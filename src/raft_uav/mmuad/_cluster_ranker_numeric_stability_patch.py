"""Keep cluster-ranker distances and normalization stable for finite extremes."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import math
from typing import Any

import numpy as np
import pandas as pd

_LABEL_PATCH_MARKER = "_raft_uav_cluster_ranker_numeric_stability"
_STANDARDIZE_PATCH_MARKER = "_raft_uav_cluster_ranker_standardize_numeric_stability"


def _scaled_mean(values: np.ndarray) -> float:
    """Return a finite mean without requiring the unscaled sum to be finite."""

    if values.size == 0:
        return 0.0
    scale = max(abs(float(value)) for value in values)
    if scale == 0.0:
        return 0.0
    normalized = math.fsum(float(value) / scale for value in values) / len(values)
    normalized = min(1.0, max(-1.0, normalized))
    return scale * normalized


def _scaled_population_std(values: np.ndarray, mean: float) -> float:
    """Return population standard deviation without squaring large values."""

    if values.size == 0:
        return 0.0
    deviations = [float(value) - mean for value in values]
    scale = max(abs(value) for value in deviations)
    if scale == 0.0:
        return 0.0
    if not math.isfinite(scale):
        return float("inf")
    normalized_variance = math.fsum((value / scale) ** 2 for value in deviations) / len(values)
    return scale * math.sqrt(normalized_variance)


def _stable_standardize_training_matrix(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the legacy transform while repairing overflowed column statistics."""

    values = np.asarray(matrix, dtype=float)
    finite_mask = np.isfinite(values)
    counts = finite_mask.sum(axis=0)
    with np.errstate(over="ignore", invalid="ignore"):
        means = np.divide(
            np.where(finite_mask, values, 0.0).sum(axis=0),
            counts,
            out=np.zeros(values.shape[1], dtype=float),
            where=counts > 0,
        )
        filled = np.where(finite_mask, values, means)
        raw_scales = np.nanstd(filled, axis=0)
    bad_columns = (counts > 0) & (~np.isfinite(means) | ~np.isfinite(raw_scales))
    if bool(bad_columns.any()):
        means = means.copy()
        filled = filled.copy()
        raw_scales = raw_scales.copy()
        for column in np.flatnonzero(bad_columns):
            finite_values = values[finite_mask[:, column], column]
            mean = _scaled_mean(finite_values)
            means[column] = mean
            filled[:, column] = np.where(finite_mask[:, column], values[:, column], mean)
            raw_scales[column] = _scaled_population_std(filled[:, column], mean)
    scales = np.where(
        np.isfinite(raw_scales) & (raw_scales > 1.0e-9),
        raw_scales,
        1.0,
    )
    return filled, means, scales


def _needs_distance_repair(legacy: float, stable: float) -> bool:
    if math.isfinite(stable) and not math.isfinite(legacy):
        return True
    return legacy == 0.0 and stable > 0.0


def _repair_truth_distances(
    labeled: pd.DataFrame,
    features: pd.DataFrame,
    truth: pd.DataFrame,
    cluster_ranker: Any,
    *,
    good_threshold_m: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Repair only norm overflow/underflow while preserving ordinary outputs."""

    if labeled.empty or "truth_matched" not in labeled.columns:
        return labeled
    authoritative = cluster_ranker._authoritative_truth_rows(truth)
    if authoritative.empty:
        return labeled
    truth_by_sequence = {
        str(sequence_id): group.sort_values("time_s").reset_index(drop=True)
        for sequence_id, group in authoritative.groupby("sequence_id", sort=True)
    }
    matched = labeled["truth_matched"].fillna(False).astype(bool).to_numpy()
    legacy_3d = pd.to_numeric(labeled["truth_distance_3d_m"], errors="coerce").to_numpy(float)
    legacy_2d = pd.to_numeric(labeled["truth_distance_2d_m"], errors="coerce").to_numpy(float)
    feature_rows = pd.DataFrame(features)
    time_gate = float(max_truth_time_delta_s)
    distance_gate = float(good_threshold_m)
    thresholds = {
        "good_cluster_2m": 2.0,
        "good_cluster_5m": 5.0,
        "good_cluster_10m": 10.0,
        "good_cluster_20m": 20.0,
        "good_cluster": distance_gate,
    }

    for position in np.flatnonzero(matched):
        row = feature_rows.iloc[int(position)]
        seq_truth = truth_by_sequence.get(str(row["sequence_id"]))
        if seq_truth is None or seq_truth.empty:
            continue
        try:
            row_time = float(row["time_s"])
            truth_times = [float(value) for value in seq_truth["time_s"].tolist()]
        except (TypeError, ValueError):
            continue
        if not math.isfinite(row_time) or not all(math.isfinite(value) for value in truth_times):
            continue
        truth_position = min(
            range(len(truth_times)),
            key=lambda index: abs(truth_times[index] - row_time),
        )
        time_delta = row_time - truth_times[truth_position]
        if abs(time_delta) > time_gate:
            continue
        truth_row = seq_truth.iloc[truth_position]
        try:
            residual = tuple(
                float(row[column]) - float(truth_row[column])
                for column in ("x_m", "y_m", "z_m")
            )
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in residual):
            continue
        stable_2d = math.hypot(residual[0], residual[1])
        stable_3d = math.hypot(*residual)
        repair_2d = _needs_distance_repair(float(legacy_2d[position]), stable_2d)
        repair_3d = _needs_distance_repair(float(legacy_3d[position]), stable_3d)
        if not (repair_2d or repair_3d):
            continue
        index = int(position)
        if repair_2d:
            labeled.iloc[index, labeled.columns.get_loc("truth_distance_2d_m")] = stable_2d
        if repair_3d:
            labeled.iloc[index, labeled.columns.get_loc("truth_distance_3d_m")] = stable_3d
            if "truth_vertical_error_m" in labeled.columns:
                labeled.iloc[index, labeled.columns.get_loc("truth_vertical_error_m")] = abs(residual[2])
            for column, threshold in thresholds.items():
                if column in labeled.columns:
                    labeled.iloc[index, labeled.columns.get_loc(column)] = stable_3d <= threshold
    return labeled


def install() -> None:
    """Install finite-extreme stability repairs for cluster-ranker helpers."""

    from raft_uav.mmuad import cluster_ranker

    previous_label: Callable[..., pd.DataFrame] = cluster_ranker.label_cluster_features_against_truth
    if not getattr(previous_label, _LABEL_PATCH_MARKER, False):

        @wraps(previous_label)
        def stable_label_cluster_features_against_truth(
            features: pd.DataFrame,
            truth: pd.DataFrame,
            *,
            good_threshold_m: float = 5.0,
            max_truth_time_delta_s: float = 0.5,
        ) -> pd.DataFrame:
            with np.errstate(over="ignore", invalid="ignore"):
                labeled = previous_label(
                    features,
                    truth,
                    good_threshold_m=good_threshold_m,
                    max_truth_time_delta_s=max_truth_time_delta_s,
                )
            return _repair_truth_distances(
                labeled,
                features,
                truth,
                cluster_ranker,
                good_threshold_m=good_threshold_m,
                max_truth_time_delta_s=max_truth_time_delta_s,
            )

        setattr(stable_label_cluster_features_against_truth, _LABEL_PATCH_MARKER, True)
        setattr(stable_label_cluster_features_against_truth, "_raft_uav_previous", previous_label)
        cluster_ranker.label_cluster_features_against_truth = stable_label_cluster_features_against_truth
        cluster_ranker._IMPL.label_cluster_features_against_truth = stable_label_cluster_features_against_truth

    previous_standardize = cluster_ranker._standardize_training_matrix
    if not getattr(previous_standardize, _STANDARDIZE_PATCH_MARKER, False):

        @wraps(previous_standardize)
        def stable_standardize_training_matrix(
            matrix: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return _stable_standardize_training_matrix(matrix)

        setattr(stable_standardize_training_matrix, _STANDARDIZE_PATCH_MARKER, True)
        setattr(stable_standardize_training_matrix, "_raft_uav_previous", previous_standardize)
        cluster_ranker._standardize_training_matrix = stable_standardize_training_matrix
        cluster_ranker._IMPL._standardize_training_matrix = stable_standardize_training_matrix
