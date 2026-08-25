"""Content-addressed per-sequence cache for proposal-graph experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ._records import prediction_texts

_SCHEMA = "raft-uav-lts-proposal-sequence-cache-v1"
_SUM_FIELDS = (
    "seed_count",
    "input_proposal_rows",
    "retained_proposal_rows",
    "duplicate_suppressed_rows",
    "anchor_tracklets",
    "graph_links",
    "common_motion_steps",
    "interpolated_rows",
    "seeded_paths",
    "confirmed_birth_paths",
    "dropped_unseeded_paths",
    "output_rows",
    "output_ids",
)


def run_cached(
    arguments: Sequence[str],
    *,
    tracker_main: Callable[[list[str] | None], int],
    cache_salt: str,
    cache_dir: Path | None,
) -> int:
    """Run one tracker process per uncached sequence and materialize the result."""
    parsed, tracker_controls = _parse_arguments(arguments)
    proposal_path = parsed.proposal_path.expanduser()
    label_dir = parsed.first_frame_label_dir.expanduser()
    output_dir = parsed.output_dir.expanduser()
    output_json = (
        None if parsed.output_json is None else parsed.output_json.expanduser()
    )
    proposal_text = prediction_texts(proposal_path)
    label_paths = _label_paths(label_dir)
    available = {path.stem for path in label_paths}
    unexpected = sorted(set(proposal_text) - {f"{name}.txt" for name in available})
    if unexpected:
        raise ValueError(
            "proposal input contains unknown sequence files: " + ", ".join(unexpected)
        )
    sequences = _selected_sequences(parsed.sequences, available)
    cache_root = _cache_root(proposal_path, cache_dir)
    _guard_paths(
        proposal_path=proposal_path,
        label_dir=label_dir,
        output_dir=output_dir,
        output_json=output_json,
        cache_root=cache_root,
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, tuple[Path, dict[str, Any]]] = {}
    labels = {path.stem: path for path in label_paths}
    for sequence in sequences:
        proposal_name = f"{sequence}.txt"
        key = _cache_key(
            sequence=sequence,
            proposal_text=proposal_text.get(proposal_name, ""),
            seed_bytes=labels[sequence].read_bytes(),
            tracker_controls=tracker_controls,
            cache_salt=cache_salt,
        )
        entry = cache_root / key[:2] / key
        cached = _read_entry(entry, expected_key=key, sequence=sequence)
        if cached is None:
            cached = _generate_entry(
                entry=entry,
                key=key,
                sequence=sequence,
                proposal_text=proposal_text.get(proposal_name, ""),
                seed_path=labels[sequence],
                tracker_controls=tracker_controls,
                tracker_main=tracker_main,
            )
        artifacts[sequence] = cached

    _publish_predictions(
        {sequence: artifact[0] for sequence, artifact in artifacts.items()},
        output_dir,
    )
    summary = _aggregate_summary(
        artifacts=artifacts,
        proposal_path=proposal_path,
        label_dir=label_dir,
        output_dir=output_dir,
    )
    if output_json is not None:
        _write_json_atomic(output_json, summary)
    print(f"sequence_cache_dir={cache_root.resolve()}")
    print(f"sequence_cache_entries={len(artifacts)}")
    print(f"sequence_count={summary['sequence_count']}")
    print(f"output_rows={summary['output_rows']}")
    return 0


def _parse_arguments(
    arguments: Sequence[str],
) -> tuple[argparse.Namespace, tuple[str, ...]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("proposal_path", type=Path)
    parser.add_argument("--first-frame-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--sequences", nargs="*")
    parsed, controls = parser.parse_known_args(list(arguments))
    return parsed, tuple(controls)


def _label_paths(label_dir: Path) -> list[Path]:
    if not label_dir.exists():
        raise FileNotFoundError(
            f"first-frame label directory does not exist: {label_dir}"
        )
    if not label_dir.is_dir():
        raise NotADirectoryError(label_dir)
    labels = sorted(label_dir.glob("*.txt"))
    if not labels:
        raise ValueError(
            f"first-frame label directory contains no .txt files: {label_dir}"
        )
    return labels


def _selected_sequences(
    requested: Sequence[str] | None,
    available: set[str],
) -> tuple[str, ...]:
    names = tuple(requested or sorted(available))
    if len(names) != len(set(names)):
        raise ValueError("sequence selection contains duplicate names")
    missing = sorted(set(names) - available)
    if missing:
        raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")
    return names


def _cache_root(proposal_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser()
    proposal = proposal_path.resolve()
    return proposal.parent / ".raft-uav-lts-proposal-sequence-cache"


def _guard_paths(
    *,
    proposal_path: Path,
    label_dir: Path,
    output_dir: Path,
    output_json: Path | None,
    cache_root: Path,
) -> None:
    if not proposal_path.exists():
        raise FileNotFoundError(f"proposal input does not exist: {proposal_path}")
    proposal = proposal_path.resolve()
    labels = label_dir.resolve()
    output = output_dir.resolve()
    cache = cache_root.resolve()
    if _overlap(output, labels):
        raise ValueError("output directory must be disjoint from seed labels")
    if proposal_path.is_dir() and _overlap(output, proposal):
        raise ValueError("output directory must be disjoint from proposals")
    if proposal_path.is_file() and (output == proposal or output in proposal.parents):
        raise ValueError("output directory must be disjoint from proposals")
    if _overlap(cache, labels) or _overlap(cache, output):
        raise ValueError("sequence cache must be disjoint from labels and output")
    if proposal_path.is_dir() and _overlap(cache, proposal):
        raise ValueError("sequence cache must be disjoint from proposals")
    if proposal_path.is_file() and cache == proposal:
        raise ValueError("sequence cache must not overwrite proposals")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if cache_root.exists() and not cache_root.is_dir():
        raise NotADirectoryError(cache_root)
    if output_json is not None:
        artifact = output_json.resolve()
        if artifact == proposal or artifact == labels or labels in artifact.parents:
            raise ValueError("output JSON must not overwrite an input")
        if proposal_path.is_dir() and proposal in artifact.parents:
            raise ValueError("output JSON must be disjoint from proposals")


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _cache_key(
    *,
    sequence: str,
    proposal_text: str,
    seed_bytes: bytes,
    tracker_controls: Sequence[str],
    cache_salt: str,
) -> str:
    digest = hashlib.sha256()
    _digest_part(digest, _SCHEMA.encode("utf-8"))
    _digest_part(digest, sequence.encode("utf-8"))
    _digest_part(digest, cache_salt.encode("utf-8"))
    _digest_part(
        digest,
        json.dumps(
            list(tracker_controls),
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8"),
    )
    _digest_part(digest, proposal_text.encode("utf-8"))
    _digest_part(digest, seed_bytes)
    return digest.hexdigest()


def _digest_part(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)


def _read_entry(
    entry: Path,
    *,
    expected_key: str,
    sequence: str,
) -> tuple[Path, dict[str, Any]] | None:
    manifest_path = entry / "manifest.json"
    prediction_path = entry / f"{sequence}.txt"
    summary_path = entry / "summary.json"
    if not manifest_path.is_file() or not prediction_path.is_file():
        return None
    if not summary_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if manifest.get("schema") != _SCHEMA or manifest.get("key") != expected_key:
        return None
    if manifest.get("sequence") != sequence:
        return None
    expected_digest = manifest.get("prediction_sha256")
    if not isinstance(expected_digest, str):
        return None
    if hashlib.sha256(prediction_path.read_bytes()).hexdigest() != expected_digest:
        return None
    if summary.get("sequence_count") != 1:
        return None
    rows = summary.get("sequences")
    if not isinstance(rows, list) or len(rows) != 1:
        return None
    if rows[0].get("sequence") != sequence:
        return None
    return prediction_path, summary


def _generate_entry(
    *,
    entry: Path,
    key: str,
    sequence: str,
    proposal_text: str,
    seed_path: Path,
    tracker_controls: Sequence[str],
    tracker_main: Callable[[list[str] | None], int],
) -> tuple[Path, dict[str, Any]]:
    entry.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{sequence}-", dir=entry.parent
    ) as temporary:
        root = Path(temporary)
        proposals = root / "proposals"
        seeds = root / "seeds"
        predictions = root / "predictions"
        summary_path = root / "summary.json"
        proposals.mkdir()
        seeds.mkdir()
        (proposals / f"{sequence}.txt").write_text(
            proposal_text,
            encoding="utf-8",
        )
        shutil.copyfile(seed_path, seeds / f"{sequence}.txt")
        tracker_arguments = [
            str(proposals),
            "--first-frame-label-dir",
            str(seeds),
            "--output-dir",
            str(predictions),
            "--output-json",
            str(summary_path),
            "--sequences",
            sequence,
            *tracker_controls,
        ]
        return_code = tracker_main(tracker_arguments)
        if return_code != 0:
            raise RuntimeError(
                f"proposal tracker failed for {sequence} with code {return_code}"
            )
        source_prediction = predictions / f"{sequence}.txt"
        if not source_prediction.is_file() or not summary_path.is_file():
            raise RuntimeError(f"proposal tracker omitted artifacts for {sequence}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = source_prediction.read_bytes()
        staging = entry.parent / f".{entry.name}.stage-{uuid.uuid4().hex}"
        staging.mkdir()
        try:
            (staging / f"{sequence}.txt").write_bytes(payload)
            (staging / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            (staging / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": _SCHEMA,
                        "key": key,
                        "sequence": sequence,
                        "prediction_sha256": hashlib.sha256(payload).hexdigest(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if entry.exists():
                existing = _read_entry(
                    entry,
                    expected_key=key,
                    sequence=sequence,
                )
                if existing is not None:
                    return existing
                shutil.rmtree(entry)
            os.replace(staging, entry)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    cached = _read_entry(entry, expected_key=key, sequence=sequence)
    if cached is None:
        raise RuntimeError(f"failed to publish sequence cache entry for {sequence}")
    return cached


def _publish_predictions(
    artifacts: dict[str, Path],
    output_dir: Path,
) -> None:
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.stage-{uuid.uuid4().hex}"
    backup = output.parent / f".{output.name}.backup-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for sequence, source in sorted(artifacts.items()):
            shutil.copyfile(source, staging / f"{sequence}.txt")
        had_output = output.exists()
        if had_output:
            os.replace(output, backup)
        try:
            os.replace(staging, output)
        except Exception:
            if had_output and backup.exists() and not output.exists():
                os.replace(backup, output)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup.exists() and output.exists():
            shutil.rmtree(backup)


def _aggregate_summary(
    *,
    artifacts: dict[str, tuple[Path, dict[str, Any]]],
    proposal_path: Path,
    label_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    summaries = [artifacts[name][1] for name in artifacts]
    if not summaries:
        raise ValueError("sequence cache cannot materialize an empty selection")
    first = summaries[0]
    result: dict[str, Any] = {
        "schema": first.get("schema"),
        "proposal_path": str(proposal_path),
        "first_frame_label_dir": str(label_dir),
        "output_dir": str(output_dir),
        "parameters": first.get("parameters"),
        "sequence_count": len(summaries),
    }
    for field in _SUM_FIELDS:
        result[field] = sum(int(summary.get(field, 0)) for summary in summaries)
    sequence_rows: list[dict[str, Any]] = []
    for summary in summaries:
        if summary.get("parameters") != result["parameters"]:
            raise RuntimeError("cached sequence parameters do not match")
        rows = summary.get("sequences")
        if not isinstance(rows, list) or len(rows) != 1:
            raise RuntimeError("cached sequence summary has an invalid shape")
        sequence_rows.extend(rows)
    result["sequences"] = sequence_rows
    return result


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
