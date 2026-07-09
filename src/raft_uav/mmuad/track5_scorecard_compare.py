"""Compare MMUAD Track 5 scorecard artifacts.

The local Track 5 workflow now produces many candidate ZIPs/scorecards. This
helper turns scorecard JSON files into one ranked table, and can also compare
``pose_by_sequence`` scorecard CSVs so public-validation diagnostics can show
which sequences improved or regressed without hand-copying metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_POSE_REFERENCE_MSE = 56.88
DEFAULT_TOP3_REFERENCE_MSE = 24.51
DEFAULT_CLASS_REFERENCE_ACCURACY = 0.322
POSE_BY_SEQUENCE_METRIC_COLUMNS = ("mse", "rmse", "mean_3d", "median_3d", "p95_3d", "max_3d")


def compare_track5_scorecards(
    scorecard_paths: Iterable[Path],
    *,
    pose_reference_mse: float = DEFAULT_POSE_REFERENCE_MSE,
    top3_reference_mse: float = DEFAULT_TOP3_REFERENCE_MSE,
    class_reference_accuracy: float = DEFAULT_CLASS_REFERENCE_ACCURACY,
) -> pd.DataFrame:
    """Return a ranked table for Track 5 local scorecard JSON files."""

    records = [
        _scorecard_record(
            Path(path),
            pose_reference_mse=pose_reference_mse,
            top3_reference_mse=top3_reference_mse,
            class_reference_accuracy=class_reference_accuracy,
        )
        for path in scorecard_paths
    ]
    if not records:
        raise ValueError("at least one scorecard path is required")
    table = pd.DataFrame.from_records(records)
    for column in (
        "pose_mse_loss_m2",
        "public_rmse_3d_m",
        "public_p95_3d_m",
        "public_max_3d_m",
        "uav_type_accuracy",
    ):
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table = table.sort_values(
        ["pose_mse_loss_m2", "public_p95_3d_m", "public_max_3d_m", "scorecard_label"],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1, dtype=int))
    best_mse = table["pose_mse_loss_m2"].iloc[0]
    table["pose_mse_delta_to_best"] = table["pose_mse_loss_m2"] - best_mse
    table["pose_mse_ratio_to_best"] = table["pose_mse_loss_m2"] / best_mse
    return table


def build_pairwise_scorecard_deltas(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise metric deltas against the ranked best row."""

    table = pd.DataFrame(comparison).copy()
    if table.empty:
        return pd.DataFrame(columns=_pairwise_columns())
    best = table.sort_values("rank").iloc[0]
    records: list[dict[str, Any]] = []
    for _, row in table.sort_values("rank").iterrows():
        records.append(
            {
                "baseline_label": str(best["scorecard_label"]),
                "comparison_label": str(row["scorecard_label"]),
                "comparison_rank": int(row["rank"]),
                "pose_mse_delta": _numeric(row.get("pose_mse_loss_m2"))
                - _numeric(best.get("pose_mse_loss_m2")),
                "rmse_delta_m": _numeric(row.get("public_rmse_3d_m"))
                - _numeric(best.get("public_rmse_3d_m")),
                "p95_delta_m": _numeric(row.get("public_p95_3d_m"))
                - _numeric(best.get("public_p95_3d_m")),
                "max_delta_m": _numeric(row.get("public_max_3d_m"))
                - _numeric(best.get("public_max_3d_m")),
                "class_accuracy_delta": _numeric(row.get("uav_type_accuracy"))
                - _numeric(best.get("uav_type_accuracy")),
            }
        )
    return pd.DataFrame.from_records(records, columns=_pairwise_columns())


def compare_pose_by_sequence_tables(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    regression_tolerance_mse: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return per-sequence deltas between two scorecard pose-by-sequence tables.

    Negative delta values mean the candidate improved over the baseline.  The
    weighted pooled MSE uses the candidate per-sequence ``count`` column, so it
    matches the public Track 5 pose aggregation when both tables cover the same
    timestamp grid.
    """

    base = _normalize_pose_by_sequence_table(baseline, label=baseline_label)
    cand = _normalize_pose_by_sequence_table(candidate, label=candidate_label)
    merged = cand.merge(
        base,
        on="sequence_id",
        how="outer",
        suffixes=(f"_{candidate_label}", f"_{baseline_label}"),
        indicator=True,
    )
    merged["sequence"] = merged["sequence_id"].astype(str)
    merged["matched_in_both"] = merged["_merge"].eq("both")
    for column in POSE_BY_SEQUENCE_METRIC_COLUMNS:
        candidate_column = f"{column}_{candidate_label}"
        baseline_column = f"{column}_{baseline_label}"
        merged[f"delta_{column}"] = merged[candidate_column] - merged[baseline_column]
        with np.errstate(divide="ignore", invalid="ignore"):
            merged[f"relative_delta_{column}"] = (
                merged[f"delta_{column}"] / merged[baseline_column]
            )
    if f"count_{candidate_label}" in merged.columns and f"count_{baseline_label}" in merged.columns:
        merged["count_delta"] = merged[f"count_{candidate_label}"] - merged[
            f"count_{baseline_label}"
        ]
    ordered = merged.sort_values(
        ["matched_in_both", "delta_mse", "sequence_id"],
        ascending=[False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return ordered, _pose_by_sequence_delta_summary(
        ordered,
        baseline_label=baseline_label,
        candidate_label=candidate_label,
        regression_tolerance_mse=float(regression_tolerance_mse),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-scorecard-compare",
        description="rank local MMUAD Track 5 scorecard JSON files and optional per-sequence deltas",
    )
    parser.add_argument("scorecard_json", nargs="*", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--pairwise-delta-csv", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--pose-reference-mse", type=float, default=DEFAULT_POSE_REFERENCE_MSE)
    parser.add_argument("--top3-reference-mse", type=float, default=DEFAULT_TOP3_REFERENCE_MSE)
    parser.add_argument(
        "--class-reference-accuracy",
        type=float,
        default=DEFAULT_CLASS_REFERENCE_ACCURACY,
    )
    parser.add_argument("--baseline-pose-by-sequence-csv", type=Path)
    parser.add_argument("--candidate-pose-by-sequence-csv", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--pose-delta-csv", type=Path)
    parser.add_argument("--pose-delta-summary-json", type=Path)
    parser.add_argument("--regression-tolerance-mse", type=float, default=0.0)
    parser.add_argument(
        "--require-no-pose-regressions",
        action="store_true",
        help="exit nonzero when any common sequence worsens beyond --regression-tolerance-mse",
    )
    args = parser.parse_args(argv)

    paths: dict[str, str] = {}
    comparison = pd.DataFrame()
    if args.scorecard_json:
        if args.output_csv is None:
            parser.error("--output-csv is required when scorecard JSON files are supplied")
        comparison = compare_track5_scorecards(
            args.scorecard_json,
            pose_reference_mse=args.pose_reference_mse,
            top3_reference_mse=args.top3_reference_mse,
            class_reference_accuracy=args.class_reference_accuracy,
        )
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(args.output_csv, index=False)
        paths["comparison_csv"] = str(args.output_csv)
        if args.pairwise_delta_csv is not None:
            pairwise = build_pairwise_scorecard_deltas(comparison)
            args.pairwise_delta_csv.parent.mkdir(parents=True, exist_ok=True)
            pairwise.to_csv(args.pairwise_delta_csv, index=False)
            paths["pairwise_delta_csv"] = str(args.pairwise_delta_csv)
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(
                json.dumps(_comparison_summary(comparison, paths=paths), indent=2),
                encoding="utf-8",
            )
            paths["summary_json"] = str(args.summary_json)

    pose_delta_summary: dict[str, Any] | None = None
    if args.baseline_pose_by_sequence_csv is not None or args.candidate_pose_by_sequence_csv is not None:
        if args.baseline_pose_by_sequence_csv is None or args.candidate_pose_by_sequence_csv is None:
            parser.error("pass both --baseline-pose-by-sequence-csv and --candidate-pose-by-sequence-csv")
        if args.pose_delta_csv is None:
            parser.error("--pose-delta-csv is required for pose-by-sequence comparison")
        pose_delta, pose_delta_summary = compare_pose_by_sequence_tables(
            pd.read_csv(args.baseline_pose_by_sequence_csv),
            pd.read_csv(args.candidate_pose_by_sequence_csv),
            baseline_label=str(args.baseline_label),
            candidate_label=str(args.candidate_label),
            regression_tolerance_mse=float(args.regression_tolerance_mse),
        )
        args.pose_delta_csv.parent.mkdir(parents=True, exist_ok=True)
        pose_delta.to_csv(args.pose_delta_csv, index=False)
        paths["pose_delta_csv"] = str(args.pose_delta_csv)
        pose_summary_path = args.pose_delta_summary_json
        if pose_summary_path is not None:
            pose_summary_path.parent.mkdir(parents=True, exist_ok=True)
            pose_summary_path.write_text(
                json.dumps(_jsonable(pose_delta_summary), indent=2),
                encoding="utf-8",
            )
            paths["pose_delta_summary_json"] = str(pose_summary_path)

    if not args.scorecard_json and pose_delta_summary is None:
        parser.error("provide scorecard JSON files and/or pose-by-sequence CSVs to compare")

    print("mmuad_track5_scorecard_compare=ok")
    if not comparison.empty:
        print(f"scorecard_count={len(comparison)}")
        print(f"best_label={comparison.iloc[0]['scorecard_label']}")
        print(f"best_pose_mse={comparison.iloc[0]['pose_mse_loss_m2']}")
    if pose_delta_summary is not None:
        print(f"pose_common_sequence_count={pose_delta_summary['common_sequence_count']}")
        print(f"pose_weighted_delta_mse={pose_delta_summary['weighted_delta_mse']}")
        print(f"pose_regressed_sequence_count={pose_delta_summary['regressed_sequence_count']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    if (
        args.require_no_pose_regressions
        and pose_delta_summary is not None
        and int(pose_delta_summary["regressed_sequence_count"]) > 0
    ):
        raise SystemExit(
            "candidate has per-sequence pose regressions; "
            f"worst={pose_delta_summary['worst_regression_sequence']} "
            f"delta_mse={pose_delta_summary['worst_regression_delta_mse']}"
        )
    return 0


def _scorecard_record(
    path: Path,
    *,
    pose_reference_mse: float,
    top3_reference_mse: float,
    class_reference_accuracy: float,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    public = payload.get("public_track5", {}) if isinstance(payload, dict) else {}
    pooled = public.get("pooled", {}) if isinstance(public, dict) else {}
    validation = payload.get("validation", {}) if isinstance(payload, dict) else {}
    pose_mse = _numeric(
        pooled.get("pose_mse_loss_m2", pooled.get("mse_3d_m2", pooled.get("mse")))
    )
    class_acc = _numeric(
        pooled.get("uav_type_accuracy", pooled.get("classification_accuracy"))
    )
    return {
        "scorecard_label": _scorecard_label(path, payload),
        "scorecard_json": str(path),
        "results_path": payload.get("results_path"),
        "scorecard_leaderboard_ready": payload.get("scorecard_leaderboard_ready"),
        "codabench_upload_ready": payload.get("codabench_upload_ready"),
        "validation_leaderboard_ready": validation.get("leaderboard_ready"),
        "pose_mse_loss_m2": pose_mse,
        "public_rmse_3d_m": _numeric(pooled.get("rmse_3d_m")),
        "public_p95_3d_m": _numeric(pooled.get("p95_3d_m")),
        "public_max_3d_m": _numeric(pooled.get("max_3d_m")),
        "uav_type_accuracy": class_acc,
        "pose_mse_delta_to_reference": pose_mse - float(pose_reference_mse),
        "pose_mse_delta_to_top3_reference": pose_mse - float(top3_reference_mse),
        "class_accuracy_delta_to_reference": class_acc - float(class_reference_accuracy),
        "beats_pose_reference": bool(np.isfinite(pose_mse) and pose_mse < pose_reference_mse),
        "beats_top3_pose_reference": bool(np.isfinite(pose_mse) and pose_mse < top3_reference_mse),
        "beats_class_reference": bool(
            np.isfinite(class_acc) and class_acc > class_reference_accuracy
        ),
        "leaderboard_blocking_reasons": ";".join(
            str(item) for item in payload.get("leaderboard_blocking_reasons", [])
        ),
    }


def _normalize_pose_by_sequence_table(rows: pd.DataFrame, *, label: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return pd.DataFrame(columns=["sequence_id", "count", *POSE_BY_SEQUENCE_METRIC_COLUMNS])
    if "sequence_id" not in frame.columns and "sequence" in frame.columns:
        frame["sequence_id"] = frame["sequence"]
    if "sequence_id" not in frame.columns:
        raise ValueError(f"{label} pose table missing sequence_id/sequence column")
    if "count" not in frame.columns:
        frame["count"] = 1
    for column in ("count", *POSE_BY_SEQUENCE_METRIC_COLUMNS):
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["sequence_id"] = frame["sequence_id"].astype(str)
    return frame[["sequence_id", "count", *POSE_BY_SEQUENCE_METRIC_COLUMNS]].drop_duplicates(
        "sequence_id"
    )


def _pose_by_sequence_delta_summary(
    rows: pd.DataFrame,
    *,
    baseline_label: str,
    candidate_label: str,
    regression_tolerance_mse: float,
) -> dict[str, Any]:
    common = rows.loc[rows["matched_in_both"].astype(bool)].copy()
    candidate_only = rows.loc[rows["_merge"].eq("left_only")]
    baseline_only = rows.loc[rows["_merge"].eq("right_only")]
    if common.empty:
        return {
            "schema": "raft-uav-mmuad-track5-pose-by-sequence-comparison-v1",
            "baseline_label": baseline_label,
            "candidate_label": candidate_label,
            "common_sequence_count": 0,
            "candidate_only_sequence_count": int(len(candidate_only)),
            "baseline_only_sequence_count": int(len(baseline_only)),
            "weighted_delta_mse": None,
            "regressed_sequence_count": 0,
        }
    weights = pd.to_numeric(common[f"count_{candidate_label}"], errors="coerce").fillna(1.0)
    weights = weights.where(weights > 0.0, 1.0)
    baseline_mse = pd.to_numeric(common[f"mse_{baseline_label}"], errors="coerce")
    candidate_mse = pd.to_numeric(common[f"mse_{candidate_label}"], errors="coerce")
    delta_mse = pd.to_numeric(common["delta_mse"], errors="coerce")
    finite_delta = delta_mse[np.isfinite(delta_mse)]
    improved = finite_delta < -float(regression_tolerance_mse)
    regressed = finite_delta > float(regression_tolerance_mse)
    best_improvement = common.loc[delta_mse.idxmin()] if not finite_delta.empty else None
    worst_regression = common.loc[delta_mse.idxmax()] if not finite_delta.empty else None
    return {
        "schema": "raft-uav-mmuad-track5-pose-by-sequence-comparison-v1",
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "common_sequence_count": int(len(common)),
        "candidate_only_sequence_count": int(len(candidate_only)),
        "baseline_only_sequence_count": int(len(baseline_only)),
        "candidate_weighted_mse": _weighted_mean(candidate_mse, weights),
        "baseline_weighted_mse": _weighted_mean(baseline_mse, weights),
        "weighted_delta_mse": _weighted_mean(delta_mse, weights),
        "mean_delta_mse": _series_mean(delta_mse),
        "median_delta_mse": _series_quantile(delta_mse, 0.5),
        "p95_delta_mse": _series_quantile(delta_mse, 0.95),
        "improved_sequence_count": int(improved.sum()),
        "regressed_sequence_count": int(regressed.sum()),
        "unchanged_sequence_count": int(len(finite_delta) - improved.sum() - regressed.sum()),
        "best_improvement_sequence": None if best_improvement is None else str(best_improvement["sequence_id"]),
        "best_improvement_delta_mse": None if best_improvement is None else float(best_improvement["delta_mse"]),
        "worst_regression_sequence": None if worst_regression is None else str(worst_regression["sequence_id"]),
        "worst_regression_delta_mse": None if worst_regression is None else float(worst_regression["delta_mse"]),
        "regression_tolerance_mse": float(regression_tolerance_mse),
    }


def _scorecard_label(path: Path, payload: dict[str, Any]) -> str:
    results_path = payload.get("results_path")
    if results_path:
        return Path(str(results_path)).stem
    return Path(path).stem


def _comparison_summary(comparison: pd.DataFrame, *, paths: dict[str, str]) -> dict[str, Any]:
    best = comparison.sort_values("rank").iloc[0]
    return {
        "schema": "raft-uav-mmuad-track5-scorecard-comparison-v1",
        "scorecard_count": int(len(comparison)),
        "best_label": str(best["scorecard_label"]),
        "best_pose_mse_loss_m2": _numeric(best.get("pose_mse_loss_m2")),
        "best_public_rmse_3d_m": _numeric(best.get("public_rmse_3d_m")),
        "best_public_p95_3d_m": _numeric(best.get("public_p95_3d_m")),
        "best_public_max_3d_m": _numeric(best.get("public_max_3d_m")),
        "best_uav_type_accuracy": _numeric(best.get("uav_type_accuracy")),
        "best_scorecard_leaderboard_ready": _bool_or_none(
            best.get("scorecard_leaderboard_ready")
        ),
        "best_codabench_upload_ready": _bool_or_none(best.get("codabench_upload_ready")),
        "paths": dict(paths),
    }


def _numeric(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    data = pd.to_numeric(values, errors="coerce").to_numpy(float)
    weight = pd.to_numeric(weights, errors="coerce").to_numpy(float)
    mask = np.isfinite(data) & np.isfinite(weight) & (weight > 0.0)
    if not mask.any():
        return None
    return float(np.average(data[mask], weights=weight[mask]))


def _series_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return None if finite.empty else float(finite.mean())


def _series_quantile(values: pd.Series, quantile: float) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return None if finite.empty else float(finite.quantile(float(quantile)))


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric) and np.isfinite(float(numeric)):
        return bool(float(numeric) != 0.0)

    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "0.0", "false", "f", "no", "n", "", "nan", "none", "<na>", "nat"}:
        return False
    return None


def _pairwise_columns() -> list[str]:
    return [
        "baseline_label",
        "comparison_label",
        "comparison_rank",
        "pose_mse_delta",
        "rmse_delta_m",
        "p95_delta_m",
        "max_delta_m",
        "class_accuracy_delta",
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
