"""Train-CV selection for MMUAD candidate-uncertainty calibration.

The hierarchical candidate-sigma calibration is truth-free at inference, but its
quantile, minimum group size, and shrinkage strength still need to be selected
without reading public-validation or hidden-test truth. This module performs
leave-one-sequence-out selection on training candidates and then refits one
frozen calibration on all supplied training rows.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_uncertainty_calibration import (
    CandidateSigmaCalibration,
    apply_candidate_sigma_calibration,
    fit_candidate_sigma_calibration,
    save_candidate_sigma_calibration,
)
from raft_uav.mmuad.cluster_ranker import build_cluster_feature_table
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file

SELECTION_SCHEMA = "raft-uav-mmuad-candidate-sigma-calibration-train-cv-v1"
SELECTION_METRICS = (
    "gaussian_nll_3d",
    "coverage_error",
    "mean_normalized_squared_error",
)
_DEFAULT_TARGET_QUANTILES = (0.5, 0.68, 0.8)
_DEFAULT_MIN_GROUP_ROWS = (10, 20, 50)
_DEFAULT_SHRINKAGE_ROWS = (0.0, 25.0, 50.0, 100.0)


@dataclass(frozen=True)
class SigmaCalibrationCVConfig:
    """One candidate hyperparameter setting for train-CV selection."""

    target_quantile: float
    min_group_rows: int
    shrinkage_rows: float


def select_candidate_sigma_calibration_by_sequence_cv(
    features: pd.DataFrame,
    *,
    input_sigma_column: str = "predicted_sigma_m",
    branch_column: str = "candidate_branch",
    target_quantiles: Sequence[float] = _DEFAULT_TARGET_QUANTILES,
    min_group_rows_values: Sequence[int] = _DEFAULT_MIN_GROUP_ROWS,
    shrinkage_rows_values: Sequence[float] = _DEFAULT_SHRINKAGE_ROWS,
    scale_min: float = 0.25,
    scale_max: float = 4.0,
    selection_metric: str = "gaussian_nll_3d",
    min_train_sequences: int = 1,
) -> tuple[CandidateSigmaCalibration, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select sigma-calibration hyperparameters with leave-one-sequence-out CV.

    Lower values are better for every supported selection metric. The default
    score is an isotropic 3-D Gaussian negative-log-likelihood surrogate:
    ``3 log(sigma) + 0.5 * (distance / sigma)^2``.
    """

    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"unsupported selection_metric={selection_metric!r}; "
            f"expected one of {SELECTION_METRICS}"
        )
    rows = _finite_labeled_rows(
        features,
        input_sigma_column=input_sigma_column,
    )
    sequences = sorted(rows["sequence_id"].astype(str).unique())
    if len(sequences) < 2:
        raise ValueError("sigma calibration train-CV requires at least two sequences")
    if int(min_train_sequences) < 1:
        raise ValueError("min_train_sequences must be at least 1")

    configs = _config_grid(
        target_quantiles=target_quantiles,
        min_group_rows_values=min_group_rows_values,
        shrinkage_rows_values=shrinkage_rows_values,
    )
    fold_records: list[dict[str, Any]] = []
    for config_index, config in enumerate(configs, start=1):
        for heldout_sequence in sequences:
            train_rows = rows.loc[rows["sequence_id"] != heldout_sequence].copy()
            heldout_rows = rows.loc[rows["sequence_id"] == heldout_sequence].copy()
            train_sequence_count = int(train_rows["sequence_id"].nunique())
            if train_sequence_count < int(min_train_sequences) or heldout_rows.empty:
                continue
            calibration = fit_candidate_sigma_calibration(
                train_rows,
                input_sigma_column=input_sigma_column,
                branch_column=branch_column,
                target_quantile=config.target_quantile,
                min_group_rows=config.min_group_rows,
                shrinkage_rows=config.shrinkage_rows,
                scale_min=scale_min,
                scale_max=scale_max,
            )
            calibrated = apply_candidate_sigma_calibration(
                heldout_rows,
                calibration,
                input_sigma_column=input_sigma_column,
                output_sigma_column="cv_calibrated_sigma_m",
            ).rows
            metrics = _calibration_metrics(
                calibrated,
                sigma_column="cv_calibrated_sigma_m",
                target_quantile=config.target_quantile,
            )
            if metrics["row_count"] <= 0:
                continue
            fold_records.append(
                {
                    "config_index": int(config_index),
                    **asdict(config),
                    "heldout_sequence": str(heldout_sequence),
                    "train_sequence_count": train_sequence_count,
                    **metrics,
                }
            )

    fold_rows = pd.DataFrame.from_records(fold_records)
    if fold_rows.empty:
        raise ValueError("no sigma calibration train-CV folds could be evaluated")
    summary_rows = _aggregate_fold_rows(fold_rows)
    summary_rows = summary_rows.sort_values(
        [
            selection_metric,
            "coverage_error",
            "gaussian_nll_3d",
            "target_quantile",
            "min_group_rows",
            "shrinkage_rows",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    summary_rows.insert(0, "selection_rank", np.arange(1, len(summary_rows) + 1))
    summary_rows.insert(1, "selection_metric", selection_metric)

    best = summary_rows.iloc[0]
    selected_config = SigmaCalibrationCVConfig(
        target_quantile=float(best["target_quantile"]),
        min_group_rows=int(best["min_group_rows"]),
        shrinkage_rows=float(best["shrinkage_rows"]),
    )
    final_calibration = fit_candidate_sigma_calibration(
        rows,
        input_sigma_column=input_sigma_column,
        branch_column=branch_column,
        target_quantile=selected_config.target_quantile,
        min_group_rows=selected_config.min_group_rows,
        shrinkage_rows=selected_config.shrinkage_rows,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    selection = {
        "schema": SELECTION_SCHEMA,
        "selection_protocol": "leave-one-sequence-out__final-fit-on-all-training-rows",
        "selection_metric": selection_metric,
        "sequence_count": int(len(sequences)),
        "fold_count": int(best["fold_count"]),
        "grid_size": int(len(configs)),
        "input_sigma_column": str(input_sigma_column),
        "branch_column": str(branch_column),
        "scale_min": float(scale_min),
        "scale_max": float(scale_max),
        "min_train_sequences": int(min_train_sequences),
        "selected_config": asdict(selected_config),
        "selected_metrics": {
            metric: float(best[metric])
            for metric in (
                "gaussian_nll_3d",
                "coverage_at_1sigma",
                "coverage_error",
                "mean_normalized_squared_error",
                "mean_sigma_m",
            )
        },
    }
    return final_calibration, fold_rows, summary_rows, selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-sigma-calibration-train-cv",
        description="select MMUAD candidate-sigma calibration on training sequences",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-sigma-column", default="predicted_sigma_m")
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--target-quantile", action="append", type=float, default=[])
    parser.add_argument("--min-group-rows", action="append", type=int, default=[])
    parser.add_argument("--shrinkage-rows", action="append", type=float, default=[])
    parser.add_argument("--scale-min", type=float, default=0.25)
    parser.add_argument("--scale-max", type=float, default=4.0)
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default="gaussian_nll_3d")
    parser.add_argument("--min-train-sequences", type=int, default=1)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--write-features", action="store_true")
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidates_csv)
    truth = load_evaluation_truth_file(args.truth_csv).rows
    features = build_cluster_feature_table(
        candidates,
        truth=truth,
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
    )
    calibration, fold_rows, summary_rows, selection = (
        select_candidate_sigma_calibration_by_sequence_cv(
            features,
            input_sigma_column=args.input_sigma_column,
            branch_column=args.branch_column,
            target_quantiles=tuple(args.target_quantile) or _DEFAULT_TARGET_QUANTILES,
            min_group_rows_values=tuple(args.min_group_rows) or _DEFAULT_MIN_GROUP_ROWS,
            shrinkage_rows_values=tuple(args.shrinkage_rows) or _DEFAULT_SHRINKAGE_ROWS,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            selection_metric=args.selection_metric,
            min_train_sequences=args.min_train_sequences,
        )
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_json = output_dir / "mmuad_candidate_sigma_calibration_train_selected.json"
    selection_json = output_dir / "mmuad_candidate_sigma_calibration_train_cv_selection.json"
    fold_csv = output_dir / "mmuad_candidate_sigma_calibration_train_cv_folds.csv"
    summary_csv = output_dir / "mmuad_candidate_sigma_calibration_train_cv_summary.csv"
    save_candidate_sigma_calibration(calibration, calibration_json)
    selection_json.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    fold_rows.to_csv(fold_csv, index=False)
    summary_rows.to_csv(summary_csv, index=False)
    if args.write_features:
        features.to_csv(output_dir / "mmuad_candidate_sigma_calibration_features.csv", index=False)

    print("mmuad_candidate_sigma_calibration_train_cv=ok")
    print(f"calibration_json={calibration_json}")
    print(f"selection_json={selection_json}")
    print(f"fold_csv={fold_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"selected_config={json.dumps(selection['selected_config'], sort_keys=True)}")
    return 0


def _finite_labeled_rows(
    features: pd.DataFrame,
    *,
    input_sigma_column: str,
) -> pd.DataFrame:
    rows = pd.DataFrame(features).copy()
    required = {"sequence_id", "truth_distance_3d_m", input_sigma_column}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"sigma calibration train-CV missing columns: {sorted(missing)}")
    truth = pd.to_numeric(rows["truth_distance_3d_m"], errors="coerce")
    sigma = pd.to_numeric(rows[input_sigma_column], errors="coerce")
    finite = np.isfinite(truth.to_numpy(float)) & np.isfinite(sigma.to_numpy(float))
    finite &= truth.to_numpy(float) >= 0.0
    finite &= sigma.to_numpy(float) > 0.0
    rows = rows.loc[finite].reset_index(drop=True)
    if rows.empty:
        raise ValueError("sigma calibration train-CV has no finite labeled rows")
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    return rows


def _config_grid(
    *,
    target_quantiles: Sequence[float],
    min_group_rows_values: Sequence[int],
    shrinkage_rows_values: Sequence[float],
) -> list[SigmaCalibrationCVConfig]:
    quantiles = sorted({float(value) for value in target_quantiles})
    min_rows = sorted({int(value) for value in min_group_rows_values})
    shrinkage = sorted({float(value) for value in shrinkage_rows_values})
    if not quantiles or not min_rows or not shrinkage:
        raise ValueError("sigma calibration train-CV grid dimensions must be non-empty")
    if any(not 0.0 < value <= 1.0 for value in quantiles):
        raise ValueError("target quantiles must be in (0, 1]")
    if any(value < 1 for value in min_rows):
        raise ValueError("minimum group rows values must be at least 1")
    if any(value < 0.0 for value in shrinkage):
        raise ValueError("shrinkage rows values must be non-negative")
    return [
        SigmaCalibrationCVConfig(
            target_quantile=target_quantile,
            min_group_rows=min_group_rows,
            shrinkage_rows=shrinkage_rows,
        )
        for target_quantile, min_group_rows, shrinkage_rows in product(
            quantiles,
            min_rows,
            shrinkage,
        )
    ]


def _calibration_metrics(
    rows: pd.DataFrame,
    *,
    sigma_column: str,
    target_quantile: float,
) -> dict[str, float | int]:
    truth = pd.to_numeric(rows.get("truth_distance_3d_m"), errors="coerce")
    sigma = pd.to_numeric(rows.get(sigma_column), errors="coerce")
    finite = truth.notna() & sigma.notna() & (truth >= 0.0) & (sigma > 0.0)
    if not finite.any():
        return {
            "row_count": 0,
            "gaussian_nll_3d": float("nan"),
            "coverage_at_1sigma": float("nan"),
            "coverage_error": float("nan"),
            "mean_normalized_squared_error": float("nan"),
            "mean_sigma_m": float("nan"),
        }
    truth_values = truth.loc[finite].to_numpy(float)
    sigma_values = sigma.loc[finite].to_numpy(float)
    normalized_squared = (truth_values / sigma_values) ** 2
    coverage = float(np.mean(truth_values <= sigma_values))
    return {
        "row_count": int(finite.sum()),
        "gaussian_nll_3d": float(
            np.mean(3.0 * np.log(sigma_values) + 0.5 * normalized_squared)
        ),
        "coverage_at_1sigma": coverage,
        "coverage_error": float(abs(coverage - float(target_quantile))),
        "mean_normalized_squared_error": float(np.mean(normalized_squared)),
        "mean_sigma_m": float(np.mean(sigma_values)),
    }


def _aggregate_fold_rows(fold_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "gaussian_nll_3d",
        "coverage_at_1sigma",
        "coverage_error",
        "mean_normalized_squared_error",
        "mean_sigma_m",
    )
    records: list[dict[str, Any]] = []
    group_columns = [
        "config_index",
        "target_quantile",
        "min_group_rows",
        "shrinkage_rows",
    ]
    for keys, group in fold_rows.groupby(group_columns, sort=True):
        record = dict(zip(group_columns, keys, strict=True))
        record["fold_count"] = int(len(group))
        record["row_count"] = int(pd.to_numeric(group["row_count"], errors="coerce").sum())
        for metric in metrics:
            record[metric] = _weighted_mean(group, metric)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _weighted_mean(rows: pd.DataFrame, column: str) -> float:
    values = pd.to_numeric(rows[column], errors="coerce")
    weights = pd.to_numeric(rows["row_count"], errors="coerce")
    finite = values.notna() & weights.notna() & (weights > 0.0)
    if not finite.any():
        return float("nan")
    return float(np.average(values.loc[finite].to_numpy(float), weights=weights.loc[finite]))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
