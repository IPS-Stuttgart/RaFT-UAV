"""Per-sequence cache for expensive Multi-UAV LTS proposal-graph runs."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from ._records import prediction_texts

_CACHE_SCHEMA = "raft-uav-lts-proposal-graph-sequence-cache-v1"
_TOTAL_FIELDS = (
    "sequence_count",
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
    cache_dir: Path | None = None,
) -> int:
    """Run the maintained CLI once per sequence and reuse complete outputs."""

    parsed = _parse_paths(arguments)
    proposal_texts = prediction_texts(parsed.proposal_path)
    label_paths = sorted(parsed.seed_dir.glob("*.txt"))
    if not label_paths:
        raise ValueError(
            f"first-frame label directory contains no .txt files: {parsed.seed_dir}"
        )
    expected_names = {path.name for path in label_paths}
    unexpected = sorted(set(proposal_texts) - expected_names)
    if unexpected:
        raise ValueError(
            "proposal input contains unknown sequence files: "
            + ", ".join(unexpected)
        )
    _validate_output_path(
        parsed.proposal_path,
        parsed.seed_dir,
        parsed.output_dir,
    )
    available = {path.stem for path in label_paths}
    requested = set(parsed.sequences)
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"unknown first-frame sequences: {', '.join(missing)}")
    selected = tuple(
        path for path in label_paths if not requested or path.stem in requested
    )

    root = (cache_dir or _default_cache_dir(parsed.proposal_path)).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    parsed.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in parsed.output_dir.glob("*.txt"):
        stale.unlink()

    normalized_arguments = _normalized_arguments(arguments)
    summaries: list[dict[str, object]] = []
    for label_path in selected:
        sequence = label_path.stem
        proposal_name = f"{sequence}.txt"
        proposal_text = proposal_texts.get(proposal_name, "")
        key = _cache_key(
            sequence=sequence,
            seed_bytes=label_path.read_bytes(),
            proposal_text=proposal_text,
            arguments=normalized_arguments,
            cache_salt=cache_salt,
        )
        entry = root / key
        prediction_path = entry / "prediction.txt"
        summary_path = entry / "summary.json"
        if not prediction_path.is_file() or not summary_path.is_file():
            _populate_entry(
                arguments,
                sequence=sequence,
                seed_path=label_path,
                proposal_text=proposal_text,
                entry=entry,
                tracker_main=tracker_main,
            )
        shutil.copyfile(prediction_path, parsed.output_dir / proposal_name)
        summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    if parsed.output_json is not None:
        merged = _merge_summaries(
            summaries,
            proposal_path=parsed.proposal_path,
            seed_dir=parsed.seed_dir,
            output_dir=parsed.output_dir,
        )
        parsed.output_json.parent.mkdir(parents=True, exist_ok=True)
        parsed.output_json.write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


class _ParsedPaths:
    def __init__(
        self,
        proposal_path: Path,
        seed_dir: Path,
        output_dir: Path,
        output_json: Path | None,
        sequences: tuple[str, ...],
    ) -> None:
        self.proposal_path = proposal_path
        self.seed_dir = seed_dir
        self.output_dir = output_dir
        self.output_json = output_json
        self.sequences = sequences


def _parse_paths(arguments: Sequence[str]) -> _ParsedPaths:
    if not arguments or str(arguments[0]).startswith("-"):
        raise ValueError("proposal input must be the first base-tracker argument")
    proposal_path = Path(arguments[0])
    seed_value = _option_value(arguments, "--first-frame-label-dir")
    output_value = _option_value(arguments, "--output-dir")
    if seed_value is None:
        raise ValueError("--first-frame-label-dir is required")
    if output_value is None:
        raise ValueError("--output-dir is required")
    output_json = _option_value(arguments, "--output-json")
    return _ParsedPaths(
        proposal_path,
        Path(seed_value),
        Path(output_value),
        Path(output_json) if output_json is not None else None,
        _sequence_values(arguments),
    )


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    prefix = option + "="
    for index, raw in enumerate(arguments):
        value = str(raw)
        if value.startswith(prefix):
            inline = value[len(prefix) :]
            if not inline:
                raise ValueError(f"{option} requires a value")
            return inline
        if value == option:
            if index + 1 >= len(arguments) or str(arguments[index + 1]).startswith("-"):
                raise ValueError(f"{option} requires a value")
            return str(arguments[index + 1])
    return None


def _sequence_values(arguments: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        value = str(arguments[index])
        if value == "--sequences":
            index += 1
            while index < len(arguments) and not str(arguments[index]).startswith("-"):
                result.append(str(arguments[index]))
                index += 1
            continue
        if value.startswith("--sequences="):
            inline = value.split("=", 1)[1]
            if inline:
                result.append(inline)
        index += 1
    return tuple(result)


def _validate_output_path(
    proposal_path: Path,
    seed_dir: Path,
    output_dir: Path,
) -> None:
    output = output_dir.resolve()
    seed_root = seed_dir.resolve()
    if output == seed_root or seed_root in output.parents:
        raise ValueError("output directory must not alias or be nested in seed labels")
    if proposal_path.is_dir():
        proposal_root = proposal_path.resolve()
        if output == proposal_root or proposal_root in output.parents:
            raise ValueError("output directory must not alias or be nested in proposals")


def _default_cache_dir(proposal_path: Path) -> Path:
    override = os.environ.get("RAFT_UAV_LTS_SEQUENCE_CACHE_DIR", "").strip()
    if override:
        return Path(override)
    parent = proposal_path.parent if proposal_path.parent != Path("") else Path.cwd()
    return parent / ".raft-uav-proposal-graph-sequence-cache"


def _normalized_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    normalized = list(
        _rewrite_arguments(
            arguments,
            proposal_path=Path("<proposal>"),
            seed_dir=Path("<seeds>"),
            output_dir=Path("<output>"),
            output_json=Path("<summary>"),
            sequence="<sequence>",
        )
    )
    return tuple(normalized)


def _cache_key(
    *,
    sequence: str,
    seed_bytes: bytes,
    proposal_text: str,
    arguments: tuple[str, ...],
    cache_salt: str,
) -> str:
    digest = hashlib.sha256()
    for payload in (
        _CACHE_SCHEMA.encode("utf-8"),
        sequence.encode("utf-8"),
        seed_bytes,
        proposal_text.encode("utf-8"),
        "\0".join(arguments).encode("utf-8"),
        cache_salt.encode("utf-8"),
    ):
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _populate_entry(
    arguments: Sequence[str],
    *,
    sequence: str,
    seed_path: Path,
    proposal_text: str,
    entry: Path,
    tracker_main: Callable[[list[str] | None], int],
) -> None:
    entry.mkdir(parents=True, exist_ok=True)
    work_root = Path(tempfile.mkdtemp(prefix="sequence-", dir=entry.parent))
    try:
        proposal_dir = work_root / "proposals"
        seed_dir = work_root / "seeds"
        output_dir = work_root / "predictions"
        summary_path = work_root / "summary.json"
        proposal_dir.mkdir(parents=True)
        seed_dir.mkdir(parents=True)
        (proposal_dir / f"{sequence}.txt").write_text(
            proposal_text,
            encoding="utf-8",
        )
        shutil.copyfile(seed_path, seed_dir / f"{sequence}.txt")
        rewritten = _rewrite_arguments(
            arguments,
            proposal_path=proposal_dir,
            seed_dir=seed_dir,
            output_dir=output_dir,
            output_json=summary_path,
            sequence=sequence,
        )
        status = tracker_main(list(rewritten))
        if status != 0:
            raise RuntimeError(
                f"proposal graph failed for sequence {sequence} with status {status}"
            )
        prediction = output_dir / f"{sequence}.txt"
        if not prediction.is_file() or not summary_path.is_file():
            raise RuntimeError(f"proposal graph produced incomplete output for {sequence}")
        token = uuid.uuid4().hex
        temporary_prediction = entry / f"prediction.{token}.tmp"
        temporary_summary = entry / f"summary.{token}.tmp"
        shutil.copyfile(prediction, temporary_prediction)
        shutil.copyfile(summary_path, temporary_summary)
        os.replace(temporary_prediction, entry / "prediction.txt")
        os.replace(temporary_summary, entry / "summary.json")
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


def _rewrite_arguments(
    arguments: Sequence[str],
    *,
    proposal_path: Path,
    seed_dir: Path,
    output_dir: Path,
    output_json: Path,
    sequence: str,
) -> tuple[str, ...]:
    result = [str(proposal_path)]
    index = 1
    found_seed = False
    found_output = False
    while index < len(arguments):
        value = str(arguments[index])
        if value == "--first-frame-label-dir":
            result.extend((value, str(seed_dir)))
            found_seed = True
            index += 2
            continue
        if value.startswith("--first-frame-label-dir="):
            result.append(f"--first-frame-label-dir={seed_dir}")
            found_seed = True
            index += 1
            continue
        if value == "--output-dir":
            result.extend((value, str(output_dir)))
            found_output = True
            index += 2
            continue
        if value.startswith("--output-dir="):
            result.append(f"--output-dir={output_dir}")
            found_output = True
            index += 1
            continue
        if value == "--output-json":
            index += 2
            continue
        if value.startswith("--output-json="):
            index += 1
            continue
        if value == "--sequences":
            index += 1
            while index < len(arguments) and not str(arguments[index]).startswith("-"):
                index += 1
            continue
        if value.startswith("--sequences="):
            index += 1
            continue
        result.append(value)
        index += 1
    if not found_seed:
        result.extend(("--first-frame-label-dir", str(seed_dir)))
    if not found_output:
        result.extend(("--output-dir", str(output_dir)))
    result.extend(("--output-json", str(output_json), "--sequences", sequence))
    return tuple(result)


def _merge_summaries(
    summaries: list[dict[str, object]],
    *,
    proposal_path: Path,
    seed_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    if not summaries:
        raise ValueError("sequence cache selected no sequences")
    merged = dict(summaries[0])
    parameters = merged.get("parameters")
    sequence_rows: list[object] = []
    for summary in summaries:
        if summary.get("parameters") != parameters:
            raise RuntimeError("cached proposal-graph parameters disagree")
        sequence_rows.extend(list(summary.get("sequences", [])))
    merged["proposal_path"] = str(proposal_path)
    merged["first_frame_label_dir"] = str(seed_dir)
    merged["output_dir"] = str(output_dir)
    merged["sequences"] = sequence_rows
    for field in _TOTAL_FIELDS:
        merged[field] = sum(int(summary.get(field, 0)) for summary in summaries)
    merged["sequence_count"] = len(sequence_rows)
    return merged
