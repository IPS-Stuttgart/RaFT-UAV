#!/usr/bin/env python3
"""Run a raw-controlled 102-sequence late-birth pruning sweep."""

from __future__ import annotations

import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import run_multi_uav_lts_cross_fitted_edge_evidence as crossfit
import run_multi_uav_lts_improved_evidence as improved
import run_multi_uav_lts_public_evidence as evidence
from raft_uav.multi_uav_lts._records import (
    format_detection,
    parse_detection_text,
    reject_duplicate_keys,
)
from raft_uav.multi_uav_lts.trajectory_birth_filter import (
    BirthFilterParameters,
    filter_sequence,
)

_SOURCE_NAME = "graph_delayed_translation"
_VARIANTS: tuple[tuple[str, BirthFilterParameters], ...] = (
    (
        "graph_birth_strict5",
        BirthFilterParameters(min_hits=5, min_span=4, min_mean_confidence=0.01),
    ),
    (
        "graph_birth_strict10",
        BirthFilterParameters(min_hits=10, min_span=9, min_mean_confidence=0.01),
    ),
    (
        "graph_birth_persistent20",
        BirthFilterParameters(min_hits=20, min_span=19, min_mean_confidence=0.01),
    ),
    (
        "graph_birth_border5",
        BirthFilterParameters(
            min_hits=5,
            min_span=4,
            min_mean_confidence=0.01,
            require_border_entry=True,
        ),
    ),
    (
        "graph_birth_border5_in025",
        BirthFilterParameters(
            min_hits=5,
            min_span=4,
            min_mean_confidence=0.01,
            require_border_entry=True,
            min_inward_motion=0.25,
        ),
    ),
    (
        "graph_birth_seed_only",
        BirthFilterParameters(drop_all_births=True),
    ),
)
_CANDIDATE_NAMES = (_SOURCE_NAME, *(name for name, _parameters in _VARIANTS))


def _seed_ids(path: Path) -> set[int]:
    rows = parse_detection_text(path.read_text(encoding="utf-8"), source=str(path))
    reject_duplicate_keys(rows, label="seed")
    if any(row.frame_id != 1 for row in rows):
        raise ValueError(f"{path}: expected first-frame-only labels")
    return {row.object_id for row in rows}


def _resolution_map(
    image_root: Path,
    seed_dir: Path,
) -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for dimensions, sequences in improved._sequence_resolution_groups(image_root, seed_dir):
        for sequence in sequences:
            result[sequence] = dimensions
    return result


def _materialize_variant(
    source_dir: Path,
    seed_dir: Path,
    name: str,
    parameters: BirthFilterParameters,
    resolutions: dict[str, tuple[int, int]],
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    root = run_dir / name
    output_dir = root / "predictions"
    if output_dir.exists():
        for stale in output_dir.glob("*.txt"):
            stale.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for source in sorted(source_dir.glob("*.txt")):
        sequence = source.stem
        if sequence not in resolutions:
            raise ValueError(f"missing image resolution for {sequence}")
        seed_path = seed_dir / source.name
        if not seed_path.is_file():
            raise FileNotFoundError(seed_path)
        width, height = resolutions[sequence]
        controls = replace(
            parameters,
            image_width=float(width),
            image_height=float(height),
        )
        rows = parse_detection_text(source.read_text(encoding="utf-8"), source=str(source))
        reject_duplicate_keys(rows, label="prediction")
        filtered, summary = filter_sequence(
            sequence,
            rows,
            seed_ids=_seed_ids(seed_path),
            parameters=controls,
        )
        (output_dir / source.name).write_text(
            "".join(format_detection(row) + "\n" for row in filtered),
            encoding="utf-8",
        )
        records.append(asdict(summary))

    digest, total_bytes, count = evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"{name} covers {count} sequences, expected {expected_sequences}"
        )
    evidence._write_json(
        root / "birth-filter-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-birth-filter-candidate-v1",
            "candidate": name,
            "source_candidate": _SOURCE_NAME,
            "parameters": asdict(parameters),
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            "birth_tracks": sum(int(row["birth_tracks"]) for row in records),
            "kept_birth_tracks": sum(
                int(row["kept_birth_tracks"]) for row in records
            ),
            "dropped_birth_tracks": sum(
                int(row["dropped_birth_tracks"]) for row in records
            ),
            "dropped_birth_rows": sum(
                int(row["dropped_birth_rows"]) for row in records
            ),
            "sequences": records,
        },
    )
    return output_dir


def _run_candidates(
    proposal_dir: Path,
    seed_dir: Path,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> dict[str, Path]:
    image_root = improved._image_root_from_inputs(sys.argv[1:])
    original_candidates = evidence.CANDIDATES
    evidence.CANDIDATES = ((_SOURCE_NAME, crossfit._DELAYED_COMMON_MOTION),)
    try:
        source_outputs = improved._run_candidates_with_native_dimensions(
            proposal_dir,
            seed_dir,
            image_root=image_root,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
    finally:
        evidence.CANDIDATES = original_candidates

    source_dir = source_outputs[_SOURCE_NAME]
    resolutions = _resolution_map(image_root, seed_dir)
    outputs: dict[str, Path] = {_SOURCE_NAME: source_dir}
    for name, parameters in _VARIANTS:
        outputs[name] = _materialize_variant(
            source_dir,
            seed_dir,
            name,
            parameters,
            resolutions,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
    return outputs


def main() -> int:
    evidence.CANDIDATES = tuple((name, ()) for name in _CANDIDATE_NAMES)
    evidence._baseline_cache_key = improved._proposal_source_cache_key
    evidence._prepare_baseline = improved._prepare_baseline_resumable
    evidence._run_candidates = _run_candidates
    return evidence.main()


if __name__ == "__main__":
    raise SystemExit(main())
