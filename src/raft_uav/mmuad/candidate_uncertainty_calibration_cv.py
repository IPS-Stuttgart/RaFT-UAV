"""Train-CV selection for MMUAD candidate-uncertainty calibration.

The branch/source sigma calibration is deliberately truth-free at inference,
but its quantile, minimum group size, and shrinkage strength still need to be
chosen without looking at public-validation or hidden-test truth.  This module
selects those hyperparameters with leave-one-sequence-out cross-validation on
training candidates, then refits one frozen calibration on all training rows.
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

SELECTION_SCHEMA = "raft-uav-mmuad-candidate-sigma-calibration-cv-v1"


@dataclass(frozen=True)
class SigmaCalibrationCVConfig:
    """One train-CV calibration candidate."""

    target_quantile: float
    min_group_rows: int
    shrinkage_rows: float


def select_candidate_sigma_calibration_cv(
    features: pd.DataFrame,
    *,
    input_sigma_column: str = "predicted_sigma_m",
    branch_column: str = "candidate_branch",
    target_quantiles: Sequence[float] = (0.5, 0.68, 0.8),
    min_group_rows_values: Sequence[int] = (10, 20, 50),
    shrinkage_rows_values: Sequence[float] = (0.0, 25.0, 50.0, 100.0),
    scale_min: float = 0.25,
    scale_max: float = 4.0,
    coverage_weight: float = 2.0,
    min_train_sequences: int = 2,
) -> tuple[CandidateSigmaCalibration, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select calibration hyperparameters with leave-one-sequence-out CV.

    The selection score combines an isotropic 3-D Gaussian negative-log-
    likelihood surrogate with a penalty for missing the fitted quantile's
    empirical one-sigma coverage.  Lower is better.  The selected calibration
    is finally refit on all supplied training rows.
    """

    rows = pd.DataFrame(features).copy()
    _validate_feature_rows(rows, input_sigma_column=input_sigma_column)
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    sequences = sorted(rows["sequence_id"].dropna().unique())
    if len(sequences) < 2:
        raise ValueError("sigma calibration CV requires at least two sequences")

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
            metrics = _fold_metrics(
                calibrated,
                sigma_column="cv_calibrated_sigma_m",
                target_quantile=config.target_quantile,
                coverage_weight=coverage_weight,
            )
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
        raise ValueError("no sigma calibration CV folds could be evaluated")

    summary_rows = _aggregate_cv_rows(fold_rows)
    summary_rows = summary_rows.sort_values(
        [