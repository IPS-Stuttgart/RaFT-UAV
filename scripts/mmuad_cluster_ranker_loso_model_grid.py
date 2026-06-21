"""Run a compact MMUAD cluster-ranker LOSO target/model grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.cluster_ranker import (
    _ranker_prediction_summary,
    build_cluster_feature_table,
    evaluate_cluster_ranker_loso,
    predict_cluster_scores,
    train_cluster_ranker,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.schema import CandidateFrame


GRID_ROWS: tuple[dict[str, str], ...] = (
    {"run": "good5_hgb", "target": "good5", "model": "HGB", "model_type": "hist-gradient-boosting-classifier", "target_column": "good_cluster_5m"},
    {"run": "good10_hgb", "target": "good10", "model": "HGB", "model_type": "hist-gradient-boosting-classifier", "target_column": "good_cluster_10m"},
    {"run": "good20_hgb", "target": "good20", "model": "HGB", "model_type": "hist-gradient-boosting-classifier", "target_column": "good_cluster_20m"},
    {"run": "distance_hgb", "target": "regression_distance", "model": "HGB", "model_type": "hist-gradient-boosting-regressor", "target_column": "truth_distance_3d_m"},
    {"run": "good5_rf", "target": "good5", "model": "RandomForest", "model_type": "random-forest-classifier", "target_column": "good_cluster_5m"},
    {"run": "good10_rf", "target": "good10", "model": "RandomForest", "model_type": "random-forest-classifier", "target_column": "good_cluster_10m"},
    {"run": "good20_rf", "target": "good20", "model": "RandomForest", "model_type": "random-forest-classifier", "target_column": "good_cluster_20m"},
    {"run": "distance_rf", "target": "regression_distance", "model": "RandomForest", "model_type": "random-forest-regressor", "target_column": "truth_distance_3d_m"},
    {"run": "good5_logistic", "target": "good5", "model": "logistic", "model_type": "sklearn-logistic", "target_column": "good_cluster_5m"},
    {"run": "good10_logistic", "target": "good10", "model": "logistic", "model_type": "sklearn-logistic", "target_column": "good_cluster_10m"},
    {"run": "good20_logistic", "target": "good20", "model": "logistic", "model_type": "sklearn-logistic", "target_column": "good_cluster_20m"},
    {"run": "good10_distance_hgb_ensemble", "target": "good10+distance", "model": "regression+classifier ensemble", "model_type": "hgb-ensemble", "target_column": "good_cluster_10m"},
    {"run": "good10_distance_rf_ensemble", "target": "good10+distance", "model": "regression+classifier ensemble", "model_type": "rf-ensemble", "target_column": "good_cluster_10m"},
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--truth-file", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--selection-confidence-weight", type=float, default=64.0)
    parser.add_argument("--sequence-root", type=Path)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = _load_enriched_features(args.features_csv, args.truth_file)
    enriched_path = args.output_dir / "mmuad_cluster_ranker_loso_model_grid_features.csv"
    features.to_csv(enriched_path, index=False)

    records: list[dict[str, Any]] = []
    for spec in GRID_ROWS:
        run_dir = args.output_dir / spec["run"]
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"grid_run={spec['run']}", flush=True)
        predictions, fold_summary, loso_summary = _run_loso_spec(
            features,
            spec,
            n_estimators=args.n_estimators,
            random_state=args.random_state,
        )
        predictions = _apply_composite_ranking_score(predictions)
        predictions_path = run_dir / "mmuad_cluster_ranker_loso_predictions.csv"
        folds_path = run_dir / "mmuad_cluster_ranker_loso_fold_summary.csv"
        loso_summary_path = run_dir / "mmuad_cluster_ranker_loso_summary.csv"
        predictions.to_csv(predictions_path, index=False)
        fold_summary.to_csv(folds_path, index=False)
        loso_summary.to_csv(loso_summary_path, index=False)

        tracker_status = _run_tracker(
            predictions_path,
            args.truth_file,
            args.reference_file,
            run_dir,
            selection_confidence_weight=args.selection_confidence_weight,
        )
        scorecard = _run_scorecard(
            run_dir,
            args.reference_file,
            args.sequence_root,
        )
        record = _summary_record(
            spec,
            run_dir,
            predictions_path,
            fold_summary,
            loso_summary,
            scorecard,
            tracker_status=tracker_status,
        )
        records.append(record)
        _write_grid_summary(records, args.output_dir / "mmuad_cluster_ranker_loso_model_grid.csv")
    _write_grid_summary(records, args.output_dir / "mmuad_cluster_ranker_loso_model_grid.csv")
    print(f"grid_summary_csv={args.output_dir / 'mmuad_cluster_ranker_loso_model_grid.csv'}")
    return 0


def _load_enriched_features(features_csv: Path, truth_file: Path) -> pd.DataFrame:
    raw = pd.read_csv(features_csv)
    truth = load_evaluation_truth_file(truth_file).rows
    return build_cluster_feature_table(
        CandidateFrame(raw),
        truth=truth,
        good_threshold_m=10.0,
        max_truth_time_delta_s=0.5,
    )


def _run_loso_spec(
    features: pd.DataFrame,
    spec: dict[str, str],
    *,
    n_estimators: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_type = spec["model_type"]
    if model_type == "hgb-ensemble":
        return _run_loso_ensemble(
            features,
            classifier_model_type="hist-gradient-boosting-classifier",
            regressor_model_type="hist-gradient-boosting-regressor",
            n_estimators=n_estimators,
            random_state=random_state,
        )
    if model_type == "rf-ensemble":
        return _run_loso_ensemble(
            features,
            classifier_model_type="random-forest-classifier",
            regressor_model_type="random-forest-regressor",
            n_estimators=n_estimators,
            random_state=random_state,
        )
    return evaluate_cluster_ranker_loso(
        features,
        model_type=model_type,
        target_column=spec["target_column"],
        n_estimators=n_estimators,
        random_state=random_state,
        score_distance_scale_m=10.0,
        protocol="LOSO public-validation diagnostic model grid, not submission-valid",
    )


def _run_loso_ensemble(
    features: pd.DataFrame,
    *,
    classifier_model_type: str,
    regressor_model_type: str,
    n_estimators: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = features.copy()
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    sequences = sorted(rows["sequence_id"].dropna().unique())
    predictions: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    protocol = "LOSO public-validation diagnostic model grid, not submission-valid"
    for heldout in sequences:
        train_rows = rows.loc[rows["sequence_id"] != heldout].copy()
        heldout_rows = rows.loc[rows["sequence_id"] == heldout].copy()
        classifier = train_cluster_ranker(
            train_rows,
            model_type=classifier_model_type,
            target_column="good_cluster_10m",
            n_estimators=n_estimators,
            random_state=random_state,
            score_distance_scale_m=10.0,
        )
        regressor = train_cluster_ranker(
            train_rows,
            model_type=regressor_model_type,
            target_column="truth_distance_3d_m",
            n_estimators=n_estimators,
            random_state=random_state,
            score_distance_scale_m=10.0,
        )
        classifier_score = predict_cluster_scores(heldout_rows, classifier)
        regressor_score = predict_cluster_scores(heldout_rows, regressor)
        heldout_rows["ranker_score"] = 0.5 * classifier_score + 0.5 * regressor_score
        heldout_rows["raw_confidence"] = pd.to_numeric(
            heldout_rows.get("confidence", np.nan),
            errors="coerce",
        )
        heldout_rows["confidence"] = heldout_rows["ranker_score"]
        heldout_rows["loso_heldout_sequence"] = heldout
        heldout_rows["loso_train_sequence_count"] = int(len(sequences) - 1)
        heldout_rows["loso_model_type"] = f"{classifier_model_type}+{regressor_model_type}"
        heldout_rows["loso_target_column"] = "good_cluster_10m+truth_distance_3d_m"
        heldout_rows["loso_protocol"] = protocol
        predictions.append(heldout_rows)
        fold = _ranker_prediction_summary(
            heldout_rows,
            sequence=heldout,
            split="heldout_sequence",
            protocol=protocol,
        )
        fold["train_sequence_count"] = int(len(sequences) - 1)
        fold["model_type"] = heldout_rows["loso_model_type"].iloc[0]
        fold["target_column"] = heldout_rows["loso_target_column"].iloc[0]
        folds.append(fold)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    fold_summary = pd.DataFrame.from_records(folds).sort_values("sequence_id").reset_index(drop=True)
    pooled = _ranker_prediction_summary(
        prediction_frame,
        sequence="__pooled__",
        split="pooled_loso",
        protocol=protocol,
    )
    pooled.update(
        {
            "fold_count": int(len(fold_summary)),
            "sequence_count": int(len(sequences)),
            "model_type": f"{classifier_model_type}+{regressor_model_type}",
            "target_column": "good_cluster_10m+truth_distance_3d_m",
        }
    )
    return prediction_frame, fold_summary, pd.DataFrame.from_records([pooled])


def _apply_composite_ranking_score(rows: pd.DataFrame) -> pd.DataFrame:
    out = rows.copy()
    out["ranker_score_raw"] = _numeric_column(out, "ranker_score")
    base = _frame_unit(out, "ranker_score_raw", high_good=True)
    cross = _frame_unit(out, "nearest_cross_sensor_score", high_good=True)
    neighbors = _frame_unit(out, "cross_sensor_neighbor_count", high_good=True)
    temporal = _frame_unit(out, "temporal_continuity_score", high_good=True)
    cv_penalty = _frame_unit(out, "constant_velocity_prediction_residual_m", high_good=False)
    previous = _numeric_column(out, "distance_to_previous_selected_m")
    fallback = _numeric_column(out, "prev_same_source_distance_m")
    out["_previous_distance_for_ranking"] = previous.fillna(fallback)
    previous_penalty = _frame_unit(out, "_previous_distance_for_ranking", high_good=False)
    composite = (
        base
        + 0.10 * cross
        + 0.05 * neighbors
        + 0.03 * temporal
        - 0.04 * cv_penalty
        - 0.03 * previous_penalty
    )
    out["ranking_score"] = composite
    out["ranker_score"] = composite
    out["confidence"] = composite
    out["ranking_formula"] = (
        "ranker_score +0.10*cross_sensor_score +0.05*cross_sensor_count "
        "+0.03*temporal_continuity -0.04*cv_residual -0.03*previous_distance"
    )
    return out.drop(columns=["_previous_distance_for_ranking"], errors="ignore")


def _frame_unit(rows: pd.DataFrame, column: str, *, high_good: bool) -> np.ndarray:
    values = _numeric_column(rows, column)
    out = np.zeros(len(rows), dtype=float)
    if values.empty:
        return out
    work = pd.DataFrame(
        {
            "sequence_id": rows["sequence_id"].astype(str),
            "time_s": pd.to_numeric(rows["time_s"], errors="coerce"),
            "value": values,
        },
        index=rows.index,
    )
    for _, group in work.groupby(["sequence_id", "time_s"], sort=False):
        series = pd.to_numeric(group["value"], errors="coerce")
        finite = series[np.isfinite(series)]
        if finite.empty:
            out[group.index.to_numpy()] = 0.0
            continue
        fill = float(finite.min()) if high_good else float(finite.max())
        array = series.fillna(fill).to_numpy(float)
        lo = float(np.min(array))
        hi = float(np.max(array))
        if hi <= lo:
            unit = np.full(len(array), 0.5, dtype=float)
        else:
            unit = (array - lo) / (hi - lo)
        if not high_good:
            unit = 1.0 - unit
        out[group.index.to_numpy()] = unit
    return out


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _run_tracker(
    predictions_path: Path,
    truth_file: Path,
    reference_file: Path,
    run_dir: Path,
    *,
    selection_confidence_weight: float,
) -> int:
    cmd = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.cli",
        "--candidate-csv",
        str(predictions_path),
        "--truth-file",
        str(truth_file),
        "--output-dir",
        str(run_dir),
        "--trajectory-completion-mode",
        "fixed-lag",
        "--trajectory-speed-gate-mps",
        "60",
        "--trajectory-outlier-replacement",
        "local-linear",
        "--ug2-official-complete-to-sequence-timestamps",
        "--official-validation-template-file",
        str(reference_file),
        "--ug2-official-results-csv",
        str(run_dir / "mmaud_results.csv"),
        "--selection-confidence-weight",
        str(selection_confidence_weight),
    ]
    with (run_dir / "tracker.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    return int(result.returncode)


def _run_scorecard(
    run_dir: Path,
    reference_file: Path,
    sequence_root: Path | None,
) -> dict[str, Any]:
    scorecard_json = run_dir / "track5_scorecard.json"
    cmd = [
        sys.executable,
        "-m",
        "raft_uav.mmuad.track5_scorecard_cli",
        "--results",
        str(run_dir / "mmaud_results.csv"),
        "--truth",
        str(reference_file),
        "--template",
        str(reference_file),
        "--allow-csv-submission",
        "--output-json",
        str(scorecard_json),
        "--summary-csv",
        str(run_dir / "track5_scorecard.csv"),
        "--public-evaluation-rows-csv",
        str(run_dir / "track5_public_rows.csv"),
    ]
    _ = sequence_root
    with (run_dir / "scorecard.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    if not scorecard_json.exists():
        return {}
    return json.loads(scorecard_json.read_text(encoding="utf-8"))


def _summary_record(
    spec: dict[str, str],
    run_dir: Path,
    predictions_path: Path,
    fold_summary: pd.DataFrame,
    loso_summary: pd.DataFrame,
    scorecard: dict[str, Any],
    *,
    tracker_status: int,
) -> dict[str, Any]:
    public = (scorecard.get("public_track5") or {}).get("pooled", {})
    bad_seq = fold_summary.loc[fold_summary["sequence_id"].isin(["seq0010", "seq0011", "seq0012"])]
    regret = pd.to_numeric(bad_seq.get("candidate_regret_mean_3d_m"), errors="coerce")
    pooled_loso = loso_summary.iloc[0].to_dict() if not loso_summary.empty else {}
    return {
        "run": spec["run"],
        "target": spec["target"],
        "model": spec["model"],
        "model_type": spec["model_type"],
        "target_column": spec["target_column"],
        "tracker_status": tracker_status,
        "pose_mse_loss_m2": public.get("pose_mse_loss_m2"),
        "rmse_3d_m": public.get("rmse_3d_m"),
        "p95_3d_m": public.get("p95_3d_m"),
        "max_3d_m": public.get("max_3d_m"),
        "classification_accuracy": public.get("classification_accuracy"),
        "top1_mean_3d_m": pooled_loso.get("top1_mean_3d_m"),
        "top1_p95_3d_m": pooled_loso.get("top1_p95_3d_m"),
        "candidate_regret_mean_3d_m": pooled_loso.get("candidate_regret_mean_3d_m"),
        "candidate_regret_p95_3d_m": pooled_loso.get("candidate_regret_p95_3d_m"),
        "seq0010_0011_0012_mean_regret_3d_m": float(regret.mean()) if len(regret) else np.nan,
        "seq0010_regret_mean_3d_m": _sequence_value(fold_summary, "seq0010", "candidate_regret_mean_3d_m"),
        "seq0011_regret_mean_3d_m": _sequence_value(fold_summary, "seq0011", "candidate_regret_mean_3d_m"),
        "seq0012_regret_mean_3d_m": _sequence_value(fold_summary, "seq0012", "candidate_regret_mean_3d_m"),
        "predictions_csv": str(predictions_path),
        "run_dir": str(run_dir),
    }


def _sequence_value(rows: pd.DataFrame, sequence: str, column: str) -> float:
    match = rows.loc[rows["sequence_id"] == sequence]
    if match.empty or column not in match.columns:
        return float("nan")
    value = pd.to_numeric(match[column], errors="coerce").iloc[0]
    return float(value) if np.isfinite(value) else float("nan")


def _write_grid_summary(records: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame = frame.sort_values(
            ["pose_mse_loss_m2", "p95_3d_m", "seq0010_0011_0012_mean_regret_3d_m"],
            na_position="last",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
