#!/usr/bin/env python
"""Run lower-variance MMUAD train-to-validation ranker variants.

This is an experiment runner, not a core method change.  It trains candidate
rankers on train labels, applies each model to validation candidates, optionally
runs the existing single-UAV tracker, and writes a compact train-vs-validation
feature-shift report for the ranker features actually used by the model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from raft_uav.mmuad.cluster_ranker import (  # noqa: E402
    _decode_sklearn_estimator,
    _load_candidates_from_args,
    build_cluster_feature_table,
    merge_cross_sensor_candidate_clusters,
    predict_cluster_scores,
    save_cluster_ranker_model,
    train_cluster_ranker,
    write_ranker_diagnostics,
)
from raft_uav.mmuad.completion import complete_results_to_truth_timestamps  # noqa: E402
from raft_uav.mmuad.evaluator import evaluate_mmaud_results, load_evaluation_truth_file  # noqa: E402
from raft_uav.mmuad.io import merge_candidate_frames  # noqa: E402
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns  # noqa: E402
from raft_uav.mmuad.submission import (  # noqa: E402
    estimates_to_mmaud_results_frame,
    load_official_track5_template_file,
    write_official_mmaud_results_csv,
    write_official_ug2_codabench_zip,
)
from raft_uav.mmuad.tracker import TrackerConfig, run_mmuad_tracker, write_tracker_output  # noqa: E402


@dataclass(frozen=True)
class RankerSpec:
    run: str
    model_type: str
    target_column: str
    n_estimators: int = 120
    iterations: int = 400
    learning_rate: float = 0.05
    blend_ranker_weight: float = 1.0
    blend_confidence_weight: float = 0.0


DEFAULT_SPECS: tuple[RankerSpec, ...] = (
    RankerSpec("logistic_good5", "sklearn-logistic", "good_cluster_5m", n_estimators=1),
    RankerSpec("logistic_good10", "sklearn-logistic", "good_cluster_10m", n_estimators=1),
    RankerSpec("rf_good5_lowvar", "random-forest-classifier", "good_cluster_5m", n_estimators=80),
    RankerSpec("rf_good10_lowvar", "random-forest-classifier", "good_cluster_10m", n_estimators=80),
    RankerSpec(
        "hgb_good5_regularized",
        "hist-gradient-boosting-classifier",
        "good_cluster_5m",
        n_estimators=80,
    ),
    RankerSpec(
        "hgb_good10_regularized",
        "hist-gradient-boosting-classifier",
        "good_cluster_10m",
        n_estimators=80,
    ),
    RankerSpec(
        "rf_distance_lowvar",
        "random-forest-regressor",
        "truth_distance_3d_m",
        n_estimators=80,
    ),
    RankerSpec(
        "hgb_distance_regularized",
        "hist-gradient-boosting-regressor",
        "truth_distance_3d_m",
        n_estimators=80,
    ),
    RankerSpec(
        "logistic_good10_blend50",
        "sklearn-logistic",
        "good_cluster_10m",
        n_estimators=1,
        blend_ranker_weight=0.5,
        blend_confidence_weight=0.5,
    ),
    RankerSpec(
        "rf_good10_blend50",
        "random-forest-classifier",
        "good_cluster_10m",
        n_estimators=80,
        blend_ranker_weight=0.5,
        blend_confidence_weight=0.5,
    ),
    RankerSpec(
        "hgb_good10_blend50",
        "hist-gradient-boosting-classifier",
        "good_cluster_10m",
        n_estimators=80,
        blend_ranker_weight=0.5,
        blend_confidence_weight=0.5,
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-candidates", type=Path)
    parser.add_argument("--train-sequence-root", type=Path)
    parser.add_argument("--train-truth", type=Path, required=True)
    parser.add_argument("--score-candidates", type=Path)
    parser.add_argument("--score-sequence-root", type=Path)
    parser.add_argument("--val-truth", type=Path)
    parser.add_argument("--val-template", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-glob", default="*")
    parser.add_argument("--score-sequence-glob")
    parser.add_argument("--voxel-size-m", type=float, default=0.75)
    parser.add_argument("--min-cluster-points", type=int, default=3)
    parser.add_argument("--no-apply-calibration", action="store_true")
    parser.add_argument("--good-threshold-m", type=float, default=5.0)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--cross-sensor-time-window-s", type=float, default=0.05)
    parser.add_argument("--cross-sensor-distance-gate-m", type=float, default=5.0)
    parser.add_argument("--selection-confidence-weight", type=float, default=64.0)
    parser.add_argument("--classification", default="2")
    parser.add_argument("--completion-max-interpolation-gap-s", type=float, default=1.0)
    parser.add_argument("--timestamp-tolerance-s", type=float, default=1.0e-6)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--only-run", action="append", default=[])
    parser.add_argument("--skip-tracker", action="store_true")
    args = parser.parse_args(argv)

    if args.train_candidates is None and args.train_sequence_root is None:
        raise SystemExit("provide --train-candidates or --train-sequence-root")
    if args.score_candidates is None and args.score_sequence_root is None:
        raise SystemExit("provide --score-candidates or --score-sequence-root")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    train_truth = load_evaluation_truth_file(args.train_truth)
    val_truth = load_evaluation_truth_file(args.val_truth) if args.val_truth else None
    template_rows = _load_template_rows(args.val_template, val_truth)

    train_candidates = _load_candidates_from_args(
        csv_path=args.train_candidates,
        sequence_root=args.train_sequence_root,
        sequence_glob=args.sequence_glob,
        apply_calibration=not args.no_apply_calibration,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
    )
    score_candidates = _load_candidates_from_args(
        csv_path=args.score_candidates,
        sequence_root=args.score_sequence_root,
        sequence_glob=args.score_sequence_glob or args.sequence_glob,
        apply_calibration=not args.no_apply_calibration,
        voxel_size_m=args.voxel_size_m,
        min_cluster_points=args.min_cluster_points,
    )
    score_input = merge_candidate_frames(
        [
            score_candidates,
            merge_cross_sensor_candidate_clusters(
                score_candidates,
                time_window_s=args.cross_sensor_time_window_s,
                distance_gate_m=args.cross_sensor_distance_gate_m,
            ),
        ]
    )

    train_candidates.rows.to_csv(args.output_dir / "mmuad_train_candidates.csv", index=False)
    score_input.rows.to_csv(args.output_dir / "mmuad_score_candidates_with_cross_sensor.csv", index=False)

    train_features = build_cluster_feature_table(
        train_candidates,
        truth=train_truth.rows,
        good_threshold_m=args.good_threshold_m,
        max_truth_time_delta_s=args.max_truth_time_delta_s,
        cross_sensor_time_window_s=args.cross_sensor_time_window_s,
        cross_sensor_distance_gate_m=args.cross_sensor_distance_gate_m,
    )
    score_features = build_cluster_feature_table(
        score_input,
        cross_sensor_time_window_s=args.cross_sensor_time_window_s,
        cross_sensor_distance_gate_m=args.cross_sensor_distance_gate_m,
    )
    write_ranker_diagnostics(train_features, args.output_dir / "mmuad_train_ranker_features.csv")
    write_ranker_diagnostics(score_features, args.output_dir / "mmuad_val_ranker_features.csv")

    selected_specs = _selected_specs(args.only_run)
    records: list[dict[str, Any]] = []
    feature_shift_written = False
    for spec in selected_specs:
        run_dir = runs_dir / spec.run
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"ranker_run={spec.run}", flush=True)
        started = time.time()
        model = train_cluster_ranker(
            train_features,
            model_type=spec.model_type,
            target_column=spec.target_column,
            iterations=spec.iterations,
            learning_rate=spec.learning_rate,
            random_state=args.random_state,
            n_estimators=spec.n_estimators,
            score_distance_scale_m=args.good_threshold_m,
        )
        model_path = save_cluster_ranker_model(model, run_dir / "cluster_ranker_model.json")
        scored_features = score_features.copy()
        ranker_scores = predict_cluster_scores(scored_features, model)
        scored_features["ranker_score"] = ranker_scores
        scored_candidates = _scored_candidate_frame(
            scored_features,
            ranker_scores=ranker_scores,
            ranker_weight=spec.blend_ranker_weight,
            confidence_weight=spec.blend_confidence_weight,
        )
        scored_candidates.rows.to_csv(run_dir / "mmuad_ranker_scored_candidates.csv", index=False)
        scored_features.to_csv(run_dir / "mmuad_ranker_score_features.csv", index=False)

        if not feature_shift_written:
            shift = feature_shift_frame(
                train_features,
                score_features,
                feature_columns=model.feature_columns,
                importance_by_feature=_importance_by_feature(model),
            )
            shift.to_csv(args.output_dir / "mmuad_train_val_feature_shift.csv", index=False)
            feature_shift_written = True

        record: dict[str, Any] = {
            "run": spec.run,
            "model_type": model.model_type,
            "target_column": model.target_column,
            "n_estimators": int(spec.n_estimators),
            "iterations": int(spec.iterations),
            "blend_ranker_weight": float(spec.blend_ranker_weight),
            "blend_confidence_weight": float(spec.blend_confidence_weight),
            "train_feature_rows": int(len(train_features)),
            "score_feature_rows": int(len(score_features)),
            "model_json": str(model_path),
            "scored_candidates_csv": str(run_dir / "mmuad_ranker_scored_candidates.csv"),
            "elapsed_s": float(time.time() - started),
        }
        if not args.skip_tracker:
            record.update(
                _run_tracker_and_score(
                    scored_candidates,
                    run_dir=run_dir,
                    val_truth=val_truth,
                    template_rows=template_rows,
                    args=args,
                )
            )
        records.append(record)
        _write_summary(args.output_dir, records)

    _write_summary(args.output_dir, records)
    print(f"ranker_grid_csv={args.output_dir / 'mmuad_lower_variance_train_val_ranker_grid.csv'}")
    print(f"feature_shift_csv={args.output_dir / 'mmuad_train_val_feature_shift.csv'}")
    return 0


def _selected_specs(only_run: list[str]) -> tuple[RankerSpec, ...]:
    if not only_run:
        return DEFAULT_SPECS
    requested = set(only_run)
    specs = tuple(spec for spec in DEFAULT_SPECS if spec.run in requested)
    missing = requested.difference(spec.run for spec in specs)
    if missing:
        raise SystemExit(f"unknown --only-run values: {sorted(missing)}")
    return specs


def _load_template_rows(path: Path | None, truth) -> pd.DataFrame | None:
    if path is not None:
        try:
            return load_evaluation_truth_file(path).rows
        except Exception:
            return load_official_track5_template_file(path)
    if truth is not None:
        return truth.rows
    return None


def _scored_candidate_frame(
    features: pd.DataFrame,
    *,
    ranker_scores: np.ndarray,
    ranker_weight: float,
    confidence_weight: float,
) -> CandidateFrame:
    rows = features.copy()
    raw_confidence = pd.to_numeric(rows.get("confidence", 1.0), errors="coerce").fillna(1.0)
    normalized_confidence = _minmax(raw_confidence.to_numpy(float))
    blended = float(ranker_weight) * ranker_scores + float(confidence_weight) * normalized_confidence
    denominator = max(float(ranker_weight) + float(confidence_weight), 1.0e-9)
    rows["raw_confidence"] = raw_confidence
    rows["ranker_score"] = ranker_scores
    rows["confidence"] = np.clip(blended / denominator, 0.0, 1.0)
    return CandidateFrame(normalize_candidate_columns(rows))


def _run_tracker_and_score(
    candidates: CandidateFrame,
    *,
    run_dir: Path,
    val_truth,
    template_rows: pd.DataFrame | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output = run_mmuad_tracker(
        candidates,
        val_truth,
        config=TrackerConfig(selection_confidence_weight=args.selection_confidence_weight),
    )
    paths = write_tracker_output(output, run_dir)
    out: dict[str, Any] = {f"tracker_{key}": value for key, value in paths.items()}
    legacy = estimates_to_mmaud_results_frame(output.estimates, class_name=str(args.classification))
    legacy = _force_classification(legacy, args.classification)
    legacy.to_csv(run_dir / "mmaud_results_legacy.csv", index=False)

    completed_rows = legacy
    if template_rows is not None:
        completion = complete_results_to_truth_timestamps(
            legacy,
            template_rows,
            max_interpolation_gap_s=args.completion_max_interpolation_gap_s,
            extrapolation="hold",
        )
        completed_rows = completion.rows
        completed_rows = _force_classification(completed_rows, args.classification)
        completed_rows.to_csv(run_dir / "mmaud_results_legacy_completed.csv", index=False)
        completion.diagnostics.to_csv(
            run_dir / "mmuad_official_timestamp_completion_rows.csv",
            index=False,
        )

    official_csv = run_dir / "mmaud_results.csv"
    official_zip = run_dir / "ug2_submission.zip"
    write_official_mmaud_results_csv(
        completed_rows,
        official_csv,
        classification=args.classification,
    )
    write_official_ug2_codabench_zip(
        completed_rows,
        official_zip,
        classification=args.classification,
    )
    out["mmaud_results_csv"] = str(official_csv)
    out["ug2_submission_zip"] = str(official_zip)

    if val_truth is not None:
        evaluation = evaluate_mmaud_results(
            completed_rows,
            val_truth.rows,
            metric_protocol="public-track5" if template_rows is not None else "nearest-time",
            timestamp_tolerance_s=args.timestamp_tolerance_s,
            max_time_delta_s=args.max_truth_time_delta_s,
        )
        (run_dir / "track5_scorecard_train_to_val.json").write_text(
            json.dumps(evaluation["summary"], indent=2),
            encoding="utf-8",
        )
        evaluation["rows"].to_csv(run_dir / "track5_scorecard_rows.csv", index=False)
        pooled = evaluation["summary"].get("pooled", {})
        for key in (
            "pose_mse_loss_m2",
            "rmse_3d_m",
            "mean_3d_m",
            "p95_3d_m",
            "max_3d_m",
            "classification_accuracy",
            "uav_type_accuracy",
        ):
            if key in pooled:
                out[key] = pooled[key]
        out["scorecard_leaderboard_ready"] = evaluation["summary"].get("leaderboard_ready")
    return out


def _force_classification(rows: pd.DataFrame, classification: Any) -> pd.DataFrame:
    out = rows.copy()
    out["uav_type"] = str(classification)
    out["classification"] = str(classification)
    return out


def feature_shift_frame(
    train_features: pd.DataFrame,
    val_features: pd.DataFrame,
    *,
    feature_columns: list[str],
    importance_by_feature: dict[str, float] | None = None,
) -> pd.DataFrame:
    importance_by_feature = importance_by_feature or {}
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        train_values = _feature_values(train_features, feature)
        val_values = _feature_values(val_features, feature)
        train_finite = train_values[np.isfinite(train_values)]
        val_finite = val_values[np.isfinite(val_values)]
        rows.append(
            {
                "feature": feature,
                "train_mean": _safe_mean(train_finite),
                "val_mean": _safe_mean(val_finite),
                "train_std": _safe_std(train_finite),
                "val_std": _safe_std(val_finite),
                "ks_statistic": _ks_statistic(train_finite, val_finite),
                "missing_rate_train": _missing_rate(train_values),
                "missing_rate_val": _missing_rate(val_values),
                "importance_if_available": importance_by_feature.get(feature, np.nan),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["ks_statistic", "feature"],
        ascending=[False, True],
    )


def _feature_values(rows: pd.DataFrame, feature: str) -> np.ndarray:
    if feature.startswith("source="):
        source = feature.split("=", 1)[1]
        return (
            rows.get("source", pd.Series([""] * len(rows), index=rows.index))
            .fillna("")
            .astype(str)
            .eq(source)
            .astype(float)
            .to_numpy()
        )
    if feature not in rows.columns:
        return np.full(len(rows), np.nan, dtype=float)
    return pd.to_numeric(rows[feature], errors="coerce").to_numpy(float)


def _importance_by_feature(model) -> dict[str, float]:
    columns = [str(column) for column in model.feature_columns]
    values: np.ndarray | None = None
    if model.sklearn_estimator_base64:
        estimator = _decode_sklearn_estimator(model.sklearn_estimator_base64)
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
    elif model.weights:
        values = np.abs(np.asarray(model.weights, dtype=float))
    if values is None or values.size != len(columns):
        return {}
    return {column: float(value) for column, value in zip(columns, values, strict=False)}


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values), dtype=float)
    filled = np.where(finite, values, np.nanmin(values[finite]))
    minimum = float(np.nanmin(filled))
    maximum = float(np.nanmax(filled))
    if maximum <= minimum:
        return np.full(len(values), 0.5, dtype=float)
    return (filled - minimum) / (maximum - minimum)


def _ks_statistic(left: np.ndarray, right: np.ndarray) -> float:
    left = np.sort(np.asarray(left, dtype=float))
    right = np.sort(np.asarray(right, dtype=float))
    if left.size == 0 or right.size == 0:
        return float("nan")
    values = np.sort(np.unique(np.concatenate([left, right])))
    left_cdf = np.searchsorted(left, values, side="right") / left.size
    right_cdf = np.searchsorted(right, values, side="right") / right.size
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else float("nan")


def _missing_rate(values: np.ndarray) -> float:
    return float(1.0 - np.isfinite(values).mean()) if len(values) else float("nan")


def _write_summary(output_dir: Path, records: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "mmuad_lower_variance_train_val_ranker_grid.csv", index=False)
    (output_dir / "mmuad_lower_variance_train_val_ranker_grid.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
