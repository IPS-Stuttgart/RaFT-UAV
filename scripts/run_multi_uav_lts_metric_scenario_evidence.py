#!/usr/bin/env python3
"""Add scenario routing and guarded gap completion to metric-aware LTS evidence."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import run_multi_uav_lts_metric_aware_evidence as metric
from raft_uav.multi_uav_lts.scenario_box_policy import (
    assemble_scenario_policy_predictions,
    write_summary,
)

_POLICY_NAME = "graph_metric_edge_swarm_beam_oof_scenario_policy"
_REFERENCE_PREFIX = "scenario-policy-reference"
_GAP_VARIANTS = (
    (f"{metric._METRIC_GRAPH_NAME}_rts_gap1", 1),
    (f"{metric._METRIC_GRAPH_NAME}_rts_gap2", 2),
)


def _variant_key(candidate_name: str) -> str:
    prefix = metric._METRIC_GRAPH_NAME + "_"
    return candidate_name.removeprefix(prefix)


def _materialize_gap_candidate(
    name: str,
    max_gap_frames: int,
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    source = run_dir / f"{metric._METRIC_GRAPH_NAME}_rts" / "predictions"
    if not source.is_dir():
        raise FileNotFoundError(f"RTS source for gap completion is missing: {source}")
    root = run_dir / name
    output_dir = root / "predictions"
    shutil.rmtree(output_dir, ignore_errors=True)
    metric.crossfit.evidence._run(
        [
            sys.executable,
            "-m",
            "raft_uav.multi_uav_lts.trajectory_gap_completion",
            source,
            "--output-dir",
            output_dir,
            "--output-json",
            root / "gap-completion-summary.json",
            "--max-gap-frames",
            str(max_gap_frames),
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
    digest, total_bytes, count = metric.crossfit.evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"{name} covers {count} sequences, expected {expected_sequences}"
        )
    metric.crossfit.evidence._write_json(
        root / "calibration-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-gap-candidate-v1",
            "candidate": name,
            "source_candidate": f"{metric._METRIC_GRAPH_NAME}_rts",
            "max_gap_frames": max_gap_frames,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
        },
    )
    return output_dir


def _materialize_policy(
    source_predictions: Path,
    resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
    *,
    run_dir: Path,
    expected_sequences: int,
    original_materialize,
    calibration_variants: tuple[tuple[str, tuple[str, ...]], ...],
) -> Path:
    manifest = metric.crossfit.evidence._load_json(run_dir / "proposal-source-manifest.json")
    raw_value = manifest.get("prediction_dir")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError("proposal-source manifest does not contain a raw prediction directory")
    raw_predictions = Path(raw_value).expanduser().resolve()
    if not raw_predictions.is_dir():
        raise NotADirectoryError(raw_predictions)

    truth_dir = metric.crossfit._dataset_path_from_inputs(sys.argv[1:], "truth_dir")
    seed_dir = metric.crossfit._dataset_path_from_inputs(
        sys.argv[1:],
        "first_frame_label_dir",
    )
    sequences = metric.crossfit._sequence_manifest(seed_dir, expected_sequences)
    fold_definitions = metric.crossfit._build_fold_definitions(sequences)
    folds = tuple(fold.heldout_sequences for fold in fold_definitions)

    reference_candidates: dict[str, Path] = {"none": raw_predictions}
    target_candidates: dict[str, Path] = {"none": source_predictions}
    for candidate_name, arguments in calibration_variants:
        key = _variant_key(candidate_name)
        reference_name = f"{_REFERENCE_PREFIX}-{key}"
        reference_candidates[key] = original_materialize(
            raw_predictions,
            reference_name,
            arguments,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        target_path = run_dir / candidate_name / "predictions"
        if not target_path.is_dir():
            raise FileNotFoundError(
                f"scenario policy target candidate is missing: {target_path}"
            )
        target_candidates[key] = target_path

    root = run_dir / _POLICY_NAME
    output_dir = root / "predictions"
    summary = assemble_scenario_policy_predictions(
        reference_candidates,
        target_candidates,
        truth_dir,
        folds,
        output_dir,
    )
    write_summary(summary, root / "scenario-policy-summary.json")
    digest, total_bytes, count = metric.crossfit.evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"scenario policy covers {count} sequences, expected {expected_sequences}"
        )
    metric.crossfit.evidence._write_json(
        root / "calibration-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-scenario-policy-candidate-v1",
            "candidate": _POLICY_NAME,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            "selection_reference": "truth-free raw-control calibration variants",
            "selection_truth_usage": "complementary sequences only",
            "target_predictions": "out-of-fold metric-edge graph variants",
        },
    )
    return output_dir


def main() -> int:
    original_variants = metric._CALIBRATION_VARIANTS
    original_extra_names = metric._EXTRA_NAMES
    original_materialize = metric._materialize_calibration_variant
    gap_placeholders = tuple((name, ()) for name, _max_gap in _GAP_VARIANTS)
    metric._CALIBRATION_VARIANTS = (
        *original_variants,
        *gap_placeholders,
        (_POLICY_NAME, ()),
    )
    metric._EXTRA_NAMES = (
        *original_extra_names,
        *(name for name, _max_gap in _GAP_VARIANTS),
        _POLICY_NAME,
    )
    gap_lookup = dict(_GAP_VARIANTS)

    def materialize(
        source_predictions: Path,
        name: str,
        arguments: tuple[str, ...],
        resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
        *,
        run_dir: Path,
        expected_sequences: int,
    ) -> Path:
        if name in gap_lookup:
            return _materialize_gap_candidate(
                name,
                gap_lookup[name],
                run_dir=run_dir,
                expected_sequences=expected_sequences,
            )
        if name != _POLICY_NAME:
            return original_materialize(
                source_predictions,
                name,
                arguments,
                resolution_groups,
                run_dir=run_dir,
                expected_sequences=expected_sequences,
            )
        return _materialize_policy(
            source_predictions,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
            original_materialize=original_materialize,
            calibration_variants=original_variants,
        )

    metric._materialize_calibration_variant = materialize
    try:
        return metric.main()
    finally:
        metric._CALIBRATION_VARIANTS = original_variants
        metric._EXTRA_NAMES = original_extra_names
        metric._materialize_calibration_variant = original_materialize


if __name__ == "__main__":
    raise SystemExit(main())
