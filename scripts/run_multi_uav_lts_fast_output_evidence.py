#!/usr/bin/env python3
"""Run a cheap 102-sequence sweep of raw-output calibration candidates."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import run_multi_uav_lts_improved_evidence as improved
import run_multi_uav_lts_public_evidence as evidence

_CALIBRATION_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("raw_rts", ()),
    (
        "raw_rts_u025",
        (
            "--uncertainty-scale-x",
            "0.25",
            "--uncertainty-scale-y",
            "0.25",
            "--max-area-ratio",
            "1.25",
        ),
    ),
    (
        "raw_rts_u050",
        (
            "--uncertainty-scale-x",
            "0.50",
            "--uncertainty-scale-y",
            "0.50",
            "--max-area-ratio",
            "1.50",
        ),
    ),
    (
        "raw_rts_u075",
        (
            "--uncertainty-scale-x",
            "0.75",
            "--uncertainty-scale-y",
            "0.75",
            "--max-area-ratio",
            "1.75",
        ),
    ),
    (
        "raw_rts_u100",
        (
            "--uncertainty-scale-x",
            "1.00",
            "--uncertainty-scale-y",
            "1.00",
            "--max-area-ratio",
            "2.00",
        ),
    ),
    (
        "raw_rts_u050_v010",
        (
            "--uncertainty-scale-x",
            "0.50",
            "--uncertainty-scale-y",
            "0.50",
            "--velocity-margin-scale",
            "0.10",
            "--max-area-ratio",
            "1.75",
        ),
    ),
)
_GAP_VARIANTS: tuple[tuple[str, str, int], ...] = (
    ("raw_gap1", "raw", 1),
    ("raw_gap2", "raw", 2),
    ("raw_gap5", "raw", 5),
    ("raw_gap10", "raw", 10),
    ("raw_gap18", "raw", 18),
    ("raw_rts_gap1", "raw_rts", 1),
    ("raw_rts_gap5", "raw_rts", 5),
    ("raw_rts_gap18", "raw_rts", 18),
    ("raw_rts_u050_gap1", "raw_rts_u050", 1),
    ("raw_rts_u050_gap5", "raw_rts_u050", 5),
)
_CANDIDATE_NAMES = tuple(
    [name for name, _arguments in _CALIBRATION_VARIANTS]
    + [name for name, _source, _gap in _GAP_VARIANTS]
)


def _complete_candidate(
    root: Path,
    *,
    candidate_name: str,
    expected_sequences: int,
) -> Path | None:
    output_dir = root / "predictions"
    summary_path = root / "fast-candidate-summary.json"
    if not output_dir.is_dir() or not summary_path.is_file():
        return None
    try:
        summary = evidence._load_json(summary_path)
        digest, total_bytes, count = evidence._directory_digest(output_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        summary.get("candidate") != candidate_name
        or count != expected_sequences
        or int(summary.get("sequence_count", -1)) != expected_sequences
        or total_bytes <= 0
        or int(summary.get("prediction_content_bytes", -1)) != total_bytes
        or summary.get("prediction_content_sha256") != digest
    ):
        return None
    return output_dir


def _record_candidate(
    root: Path,
    *,
    name: str,
    source_candidate: str,
    expected_sequences: int,
    controls: dict[str, Any],
) -> Path:
    output_dir = root / "predictions"
    digest, total_bytes, count = evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"{name} covers {count} sequences, expected {expected_sequences}"
        )
    evidence._write_json(
        root / "fast-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-fast-output-candidate-v1",
            "candidate": name,
            "source_candidate": source_candidate,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            **controls,
        },
    )
    return output_dir


def _materialize_calibration(
    raw_predictions: Path,
    name: str,
    arguments: tuple[str, ...],
    resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    root = run_dir / name
    complete = _complete_candidate(
        root,
        candidate_name=name,
        expected_sequences=expected_sequences,
    )
    if complete is not None:
        print(f"Reusing complete candidate {name}", flush=True)
        return complete

    output_dir = root / "predictions"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    copied: set[str] = set()
    group_records: list[dict[str, Any]] = []
    for dimensions, sequences in resolution_groups:
        width, height = dimensions
        group_root = root / "resolution-groups" / f"{width}x{height}"
        group_output = group_root / "predictions"
        shutil.rmtree(group_output, ignore_errors=True)
        evidence._run(
            [
                sys.executable,
                "-m",
                "raft_uav.multi_uav_lts.trajectory_box_calibration",
                raw_predictions,
                "--output-dir",
                group_output,
                "--output-json",
                group_root / "summary.json",
                "--image-width",
                str(width),
                "--image-height",
                str(height),
                *arguments,
                "--sequences",
                *sequences,
            ],
            log_path=group_root / "console.txt",
        )
        _, total_bytes, count = evidence._directory_digest(group_output)
        if count != len(sequences) or total_bytes <= 0:
            raise ValueError(
                f"{name}/{width}x{height} covers {count} sequences, "
                f"expected {len(sequences)}"
            )
        for source in sorted(group_output.glob("*.txt")):
            if source.name in copied:
                raise ValueError(f"{name}: duplicate prediction {source.name}")
            shutil.copy2(source, output_dir / source.name)
            copied.add(source.name)
        group_records.append(
            {
                "width": width,
                "height": height,
                "sequences": list(sequences),
                "prediction_content_bytes": total_bytes,
            }
        )
    return _record_candidate(
        root,
        name=name,
        source_candidate="raw",
        expected_sequences=expected_sequences,
        controls={
            "operation": "rts-box-calibration",
            "arguments": list(arguments),
            "resolution_groups": group_records,
        },
    )


def _materialize_gap(
    source_predictions: Path,
    name: str,
    source_name: str,
    max_gap: int,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    root = run_dir / name
    complete = _complete_candidate(
        root,
        candidate_name=name,
        expected_sequences=expected_sequences,
    )
    if complete is not None:
        print(f"Reusing complete candidate {name}", flush=True)
        return complete

    output_dir = root / "predictions"
    shutil.rmtree(output_dir, ignore_errors=True)
    evidence._run(
        [
            sys.executable,
            "-m",
            "raft_uav.multi_uav_lts.trajectory_gap_completion",
            source_predictions,
            "--output-dir",
            output_dir,
            "--output-json",
            root / "gap-completion-summary.json",
            "--max-gap-frames",
            str(max_gap),
            "--max-normalized-speed",
            "5.0",
            "--max-log-size-change",
            "1.0",
            "--min-endpoint-confidence",
            "0.003",
            "--confidence-decay",
            "0.85",
            "--raw-endpoints",
        ],
        log_path=root / "console.txt",
    )
    return _record_candidate(
        root,
        name=name,
        source_candidate=source_name,
        expected_sequences=expected_sequences,
        controls={
            "operation": "guarded-gap-completion",
            "max_gap_frames": max_gap,
        },
    )


def _run_fast_candidates(
    proposal_dir: Path,
    seed_dir: Path,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Path]:
    raw_predictions = proposal_dir.parent / "predictions"
    _, raw_bytes, raw_count = evidence._directory_digest(raw_predictions)
    if raw_count != expected_sequences or raw_bytes <= 0:
        raise ValueError(
            f"raw control covers {raw_count} sequences, expected {expected_sequences}"
        )
    image_root = improved._image_root_from_inputs(sys.argv[1:])
    resolution_groups = improved._sequence_resolution_groups(image_root, seed_dir)

    outputs: dict[str, Path] = {}
    for name, arguments in _CALIBRATION_VARIANTS:
        outputs[name] = _materialize_calibration(
            raw_predictions,
            name,
            arguments,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )

    sources = {"raw": raw_predictions, **outputs}
    for name, source_name, max_gap in _GAP_VARIANTS:
        outputs[name] = _materialize_gap(
            sources[source_name],
            name,
            source_name,
            max_gap,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
    return outputs


def main() -> int:
    evidence.CANDIDATES = tuple((name, ()) for name in _CANDIDATE_NAMES)
    evidence._baseline_cache_key = improved._proposal_source_cache_key
    evidence._prepare_baseline = improved._prepare_baseline_resumable
    evidence._run_candidates = _run_fast_candidates
    return evidence.main()


if __name__ == "__main__":
    raise SystemExit(main())
