#!/usr/bin/env python3
"""Run cross-fitted metric-edge tracking and uncertainty-aware RTS box candidates."""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import run_multi_uav_lts_cross_fitted_edge_evidence as crossfit

_METRIC_MODEL_TOKEN = "{METRIC_EDGE_MODEL_JSON}"
_METRIC_GRAPH_NAME = "graph_metric_edge_swarm_beam_oof"
_METRIC_GRAPH_ARGUMENTS = (
    *crossfit._DELAYED_COMMON_MOTION,
    *crossfit._SIMILARITY_MOTION,
    *crossfit._SWARM_RELATIVE,
    *crossfit._AMBIGUITY_BEAM,
    "--metric-edge-model-json",
    _METRIC_MODEL_TOKEN,
)
_CALIBRATION_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        f"{_METRIC_GRAPH_NAME}_rts",
        (
            "--uncertainty-scale-x",
            "0.0",
            "--uncertainty-scale-y",
            "0.0",
        ),
    ),
    (
        f"{_METRIC_GRAPH_NAME}_rts_u050",
        (
            "--uncertainty-scale-x",
            "0.5",
            "--uncertainty-scale-y",
            "0.5",
            "--max-area-ratio",
            "1.5",
        ),
    ),
    (
        f"{_METRIC_GRAPH_NAME}_rts_u100",
        (
            "--uncertainty-scale-x",
            "1.0",
            "--uncertainty-scale-y",
            "1.0",
            "--max-area-ratio",
            "2.0",
        ),
    ),
    (
        f"{_METRIC_GRAPH_NAME}_rts_u100_v015",
        (
            "--uncertainty-scale-x",
            "1.0",
            "--uncertainty-scale-y",
            "1.0",
            "--velocity-margin-scale",
            "0.15",
            "--max-area-ratio",
            "2.25",
        ),
    ),
)
_EXTRA_NAMES = (
    _METRIC_GRAPH_NAME,
    *(name for name, _arguments in _CALIBRATION_VARIANTS),
)


@dataclass(frozen=True)
class MetricFoldModel:
    fold_index: int
    model_path: Path
    summary_path: Path
    model_sha256: str
    training_sequences: tuple[str, ...]
    heldout_sequences: tuple[str, ...]


def _validate_metric_model_provenance(
    model_path: Path,
    summary_path: Path,
    fold: crossfit.FoldDefinition,
) -> MetricFoldModel:
    model = crossfit.evidence._load_json(model_path)
    summary = crossfit.evidence._load_json(summary_path)
    expected = list(fold.training_sequences)
    metadata = model.get("metadata")
    selected_model = (
        metadata.get("selected_sequences") if isinstance(metadata, Mapping) else None
    )
    if selected_model != expected or summary.get("selected_sequences") != expected:
        raise ValueError(
            f"fold {fold.index} metric-edge provenance does not match its training panel"
        )
    if set(fold.heldout_sequences) & set(fold.training_sequences):
        raise ValueError(
            f"fold {fold.index} metric-edge training panel leaks held-out data"
        )
    if int(model.get("sequence_count", 0)) != len(expected):
        raise ValueError(
            f"fold {fold.index} metric-edge sequence_count is inconsistent"
        )
    for field in (
        "identity_positive_edges",
        "hota_005_positive_edges",
        "clear_050_positive_edges",
    ):
        if int(summary.get(field, 0)) <= 0:
            raise ValueError(f"fold {fold.index} metric-edge summary has no {field}")
    return MetricFoldModel(
        fold_index=fold.index,
        model_path=model_path,
        summary_path=summary_path,
        model_sha256=crossfit.evidence._sha256(model_path),
        training_sequences=fold.training_sequences,
        heldout_sequences=fold.heldout_sequences,
    )


def _fit_metric_fold_models(
    proposal_dir: Path,
    truth_dir: Path,
    folds: tuple[crossfit.FoldDefinition, ...],
    *,
    run_dir: Path,
) -> dict[int, MetricFoldModel]:
    models: dict[int, MetricFoldModel] = {}
    for fold in folds:
        root = run_dir / "cross-fitted-metric-edge-models" / f"fold-{fold.index}"
        model_path = root / "metric-edge-model.json"
        summary_path = root / "fit-summary.json"
        crossfit.evidence._run(
            [
                sys.executable,
                "-m",
                "raft_uav.multi_uav_lts.proposal_metric_edge_model",
                proposal_dir,
                "--truth-dir",
                truth_dir,
                "--output-json",
                model_path,
                "--summary-json",
                summary_path,
                "--max-gap",
                "0",
                "--max-link-cost",
                "2.25",
                "--negative-candidates-per-left",
                "5",
                "--swarm-neighbors",
                "4",
                "--swarm-radius-scale",
                "12.0",
                "--swarm-unmatched-penalty",
                "2.0",
                "--l2-penalty",
                "1.0",
                "--identity-weight",
                "0.75",
                "--hota-weight",
                "1.0",
                "--clear-weight",
                "0.25",
                "--sequences",
                *fold.training_sequences,
            ],
            log_path=root / "fit-console.txt",
        )
        models[fold.index] = _validate_metric_model_provenance(
            model_path,
            summary_path,
            fold,
        )
    return models


def _resolve_metric_arguments(
    arguments: tuple[str, ...],
    model_path: Path,
) -> tuple[str, ...]:
    if arguments.count(_METRIC_MODEL_TOKEN) != 1:
        raise ValueError("metric candidate arguments must contain one model placeholder")
    return tuple(
        str(model_path) if value == _METRIC_MODEL_TOKEN else value
        for value in arguments
    )


def _run_metric_tracker_group(
    proposal_dir: Path,
    seed_dir: Path,
    *,
    output_dir: Path,
    summary_path: Path,
    log_path: Path,
    dimensions: tuple[int, int],
    arguments: tuple[str, ...],
    sequences: tuple[str, ...],
) -> None:
    width, height = dimensions
    crossfit.evidence._run(
        [
            sys.executable,
            "-m",
            "raft_uav.multi_uav_lts.metric_proposal_graph_tracker",
            proposal_dir,
            "--first-frame-label-dir",
            seed_dir,
            "--output-dir",
            output_dir,
            "--output-json",
            summary_path,
            "--image-width",
            str(width),
            "--image-height",
            str(height),
            *arguments,
            "--sequences",
            *sequences,
        ],
        log_path=log_path,
    )


def _materialize_metric_candidate(
    proposal_dir: Path,
    seed_dir: Path,
    folds: tuple[crossfit.FoldDefinition, ...],
    models: Mapping[int, MetricFoldModel],
    resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    root = run_dir / _METRIC_GRAPH_NAME
    output_dir = root / "predictions"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    copied: set[str] = set()
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        model = models[fold.index]
        if model.heldout_sequences != fold.heldout_sequences:
            raise ValueError(
                f"fold {fold.index} metric model was assigned to the wrong fold"
            )
        arguments = _resolve_metric_arguments(
            _METRIC_GRAPH_ARGUMENTS,
            model.model_path,
        )
        groups: list[dict[str, Any]] = []
        for dimensions, sequences in crossfit._selected_resolution_groups(
            resolution_groups,
            fold.heldout_sequences,
        ):
            width, height = dimensions
            group_root = (
                root / "fold-groups" / f"fold-{fold.index}-{width}x{height}"
            )
            group_output = group_root / "predictions"
            shutil.rmtree(group_output, ignore_errors=True)
            _run_metric_tracker_group(
                proposal_dir,
                seed_dir,
                output_dir=group_output,
                summary_path=group_root / "summary.json",
                log_path=group_root / "console.txt",
                dimensions=dimensions,
                arguments=arguments,
                sequences=sequences,
            )
            _digest, total_bytes, count = crossfit.evidence._directory_digest(
                group_output
            )
            if count != len(sequences) or total_bytes <= 0:
                raise ValueError(
                    "metric fold group produced "
                    f"{count} nonempty files for {len(sequences)} sequences"
                )
            for source in sorted(group_output.glob("*.txt")):
                if source.name in copied:
                    raise ValueError(
                        f"duplicate metric held-out prediction: {source.name}"
                    )
                shutil.copy2(source, output_dir / source.name)
                copied.add(source.name)
            groups.append(
                {
                    "width": width,
                    "height": height,
                    "sequences": list(sequences),
                    "prediction_content_bytes": total_bytes,
                }
            )
        fold_records.append(
            {
                "fold": fold.index,
                "training_sequences": list(fold.training_sequences),
                "heldout_sequences": list(fold.heldout_sequences),
                "metric_edge_model_sha256": model.model_sha256,
                "resolution_groups": groups,
            }
        )
    expected_names = {
        f"{sequence}.txt"
        for fold in folds
        for sequence in fold.heldout_sequences
    }
    if copied != expected_names:
        raise ValueError(
            "metric held-out assembly is incomplete: "
            f"missing={sorted(expected_names - copied)}, "
            f"extra={sorted(copied - expected_names)}"
        )
    digest, total_bytes, count = crossfit.evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"metric candidate covers {count} sequences, expected {expected_sequences}"
        )
    crossfit.evidence._write_json(
        root / "cross-fitted-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-cross-fitted-metric-edge-candidate-v1",
            "candidate": _METRIC_GRAPH_NAME,
            "fold_count": len(folds),
            "fold_seed": crossfit.FOLD_SEED,
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            "truth_usage": (
                "metric edge heads only; each prediction uses complementary-fold truth"
            ),
            "folds": fold_records,
        },
    )
    return output_dir


def _materialize_calibration_variant(
    source_predictions: Path,
    name: str,
    arguments: tuple[str, ...],
    resolution_groups: tuple[tuple[tuple[int, int], tuple[str, ...]], ...],
    *,
    run_dir: Path,
    expected_sequences: int,
) -> Path:
    root = run_dir / name
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
        crossfit.evidence._run(
            [
                sys.executable,
                "-m",
                "raft_uav.multi_uav_lts.trajectory_box_calibration",
                source_predictions,
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
        for source in sorted(group_output.glob("*.txt")):
            if source.name in copied:
                raise ValueError(
                    f"{name}: duplicate calibrated prediction {source.name}"
                )
            shutil.copy2(source, output_dir / source.name)
            copied.add(source.name)
        group_records.append(
            {
                "width": width,
                "height": height,
                "sequences": list(sequences),
            }
        )
    digest, total_bytes, count = crossfit.evidence._directory_digest(output_dir)
    if count != expected_sequences or total_bytes <= 0:
        raise ValueError(
            f"{name} covers {count} sequences, expected {expected_sequences}"
        )
    crossfit.evidence._write_json(
        root / "calibration-candidate-summary.json",
        {
            "schema": "raft-uav-multi-uav-lts-rts-box-candidate-v1",
            "candidate": name,
            "source_candidate": _METRIC_GRAPH_NAME,
            "arguments": list(arguments),
            "sequence_count": count,
            "prediction_content_bytes": total_bytes,
            "prediction_content_sha256": digest,
            "resolution_groups": group_records,
        },
    )
    return output_dir


def _available_results_with_metric(
    original_available,
    original_crossfit_helper,
    run_dir: Path,
) -> dict[str, Any]:
    payload = original_crossfit_helper(original_available, run_dir)
    metrics = payload.setdefault("metrics", {})
    for name in _EXTRA_NAMES:
        path = run_dir / name / "metrics.json"
        if not path.is_file():
            continue
        try:
            metrics[name] = crossfit.evidence._load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            metrics[name] = {"read_error": f"{type(exc).__name__}: {exc}"}
    return payload


def main() -> int:
    original_runner = crossfit._run_out_of_fold_candidates
    original_available_helper = crossfit._available_results_with_out_of_fold

    def run_out_of_fold_candidates(
        proposal_dir: Path,
        seed_dir: Path,
        truth_dir: Path,
        *,
        image_root: Path,
        run_dir: Path,
        expected_sequences: int,
    ) -> dict[str, Path]:
        legacy = original_runner(
            proposal_dir,
            seed_dir,
            truth_dir,
            image_root=image_root,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        sequences = crossfit._sequence_manifest(seed_dir, expected_sequences)
        folds = crossfit._build_fold_definitions(sequences)
        resolution_groups = crossfit.improved._sequence_resolution_groups(
            image_root,
            seed_dir,
        )
        metric_models = _fit_metric_fold_models(
            proposal_dir,
            truth_dir,
            folds,
            run_dir=run_dir,
        )
        metric_output = _materialize_metric_candidate(
            proposal_dir,
            seed_dir,
            folds,
            metric_models,
            resolution_groups,
            run_dir=run_dir,
            expected_sequences=expected_sequences,
        )
        derived = {
            name: _materialize_calibration_variant(
                metric_output,
                name,
                arguments,
                resolution_groups,
                run_dir=run_dir,
                expected_sequences=expected_sequences,
            )
            for name, arguments in _CALIBRATION_VARIANTS
        }
        overlap = set(legacy) & ({_METRIC_GRAPH_NAME} | set(derived))
        if overlap:
            raise ValueError(
                f"duplicate metric-aware candidate names: {sorted(overlap)}"
            )
        return {**legacy, _METRIC_GRAPH_NAME: metric_output, **derived}

    def available_results_with_metric(original_available, run_dir: Path):
        return _available_results_with_metric(
            original_available,
            original_available_helper,
            run_dir,
        )

    crossfit._run_out_of_fold_candidates = run_out_of_fold_candidates
    crossfit._available_results_with_out_of_fold = available_results_with_metric
    try:
        return crossfit.main()
    finally:
        crossfit._run_out_of_fold_candidates = original_runner
        crossfit._available_results_with_out_of_fold = original_available_helper


if __name__ == "__main__":
    raise SystemExit(main())
