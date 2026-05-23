"""Paper-style empirical RF/radar measurement covariance utilities.

The reference baseline estimates RF and radar measurement covariances directly
from sensor-minus-ground-truth residuals before NIS validation and Kalman
updates.  This module implements that deliberately simple protocol and writes
``cov_*`` columns already consumed by ``covariance_from_row`` and the radar
association code.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

_SOURCE_COORDS = {
    "rf": ("east_m", "north_m"),
    "radar": ("east_m", "north_m", "up_m"),
}
_COVARIANCE_SUFFIXES = {
    2: ((0, 0, "ee"), (1, 1, "nn"), (0, 1, "en")),
    3: (
        (0, 0, "ee"),
        (1, 1, "nn"),
        (2, 2, "uu"),
        (0, 1, "en"),
        (0, 2, "eu"),
        (1, 2, "nu"),
    ),
}


def estimate_empirical_measurement_covariances(
    *,
    rf: pd.DataFrame | None,
    radar: pd.DataFrame | None,
    truth: pd.DataFrame,
    max_time_delta_s: float = 2.0,
    min_variance_m2: float = 1.0,
) -> dict[str, Any]:
    """Estimate source-specific sample covariance matrices from residuals.

    ``np.cov(..., bias=False)`` matches MATLAB ``cov`` normalization by using
    ``N - 1`` in the denominator.  A small diagonal floor prevents singular NIS
    matrices in tiny synthetic tests or very short diagnostic slices.
    """

    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_type": "paper-empirical-measurement-covariance",
        "max_time_delta_s": float(max_time_delta_s),
        "min_variance_m2": float(min_variance_m2),
    }
    for source, frame in (("rf", rf), ("radar", radar)):
        if frame is None or frame.empty:
            continue
        residuals = aligned_residuals(
            frame,
            truth,
            source=source,
            max_time_delta_s=max_time_delta_s,
        )
        if residuals.size == 0:
            continue
        covariance = empirical_covariance_matrix(
            residuals,
            min_variance_m2=min_variance_m2,
        )
        payload[source] = {
            "dimensions": list(_SOURCE_COORDS[source]),
            "sample_count": int(residuals.shape[0]),
            "mean_residual_m": [float(v) for v in np.mean(residuals, axis=0)],
            "std_residual_m": [float(v) for v in np.std(residuals, axis=0, ddof=1)]
            if residuals.shape[0] > 1
            else [0.0] * residuals.shape[1],
            "covariance_m2": covariance.tolist(),
        }
    return payload


def aligned_residuals(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    max_time_delta_s: float,
) -> np.ndarray:
    """Return sensor-minus-nearest-truth residuals for one source."""

    if source not in _SOURCE_COORDS:
        raise ValueError(f"unknown source {source!r}")
    required = ("time_s", *_SOURCE_COORDS[source])
    if frame.empty or not all(column in frame.columns for column in required):
        return np.empty((0, len(_SOURCE_COORDS[source])), dtype=float)
    if truth.empty or not all(column in truth.columns for column in required):
        return np.empty((0, len(_SOURCE_COORDS[source])), dtype=float)

    truth_times = truth["time_s"].to_numpy(dtype=float)
    query_times = frame["time_s"].to_numpy(dtype=float)
    if truth_times.size == 0 or query_times.size == 0:
        return np.empty((0, len(_SOURCE_COORDS[source])), dtype=float)
    truth_indices = _nearest_time_indices(truth_times, query_times)
    dt_s = np.abs(truth_times[truth_indices] - query_times)
    keep = dt_s <= float(max_time_delta_s)
    if not keep.any():
        return np.empty((0, len(_SOURCE_COORDS[source])), dtype=float)

    coords = list(_SOURCE_COORDS[source])
    measured = frame.loc[keep, coords].to_numpy(dtype=float)
    reference = truth.iloc[truth_indices[keep]][coords].to_numpy(dtype=float)
    residuals = measured - reference
    residuals = residuals[np.isfinite(residuals).all(axis=1)]
    return residuals.reshape((-1, len(coords)))


def empirical_covariance_matrix(
    residuals: np.ndarray,
    *,
    min_variance_m2: float = 1.0,
) -> np.ndarray:
    """Return a finite symmetric covariance matrix with a diagonal floor."""

    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 2:
        raise ValueError("residuals must be a 2D array")
    if residuals.shape[0] == 0:
        raise ValueError("cannot estimate empirical covariance without residuals")
    residuals = residuals[np.isfinite(residuals).all(axis=1)]
    if residuals.shape[0] == 0:
        raise ValueError("cannot estimate empirical covariance without finite residuals")
    dim = residuals.shape[1]
    if residuals.shape[0] == 1:
        covariance = np.eye(dim, dtype=float) * float(min_variance_m2)
    else:
        covariance = np.asarray(np.cov(residuals, rowvar=False, bias=False), dtype=float)
        covariance = covariance.reshape((dim, dim))
    covariance = 0.5 * (covariance + covariance.T)
    covariance[~np.isfinite(covariance)] = 0.0
    floor = float(min_variance_m2)
    for index in range(dim):
        covariance[index, index] = max(float(covariance[index, index]), floor)
    return covariance


def apply_empirical_covariance(
    frame: pd.DataFrame,
    *,
    source: str,
    covariance_payload: Mapping[str, Any],
) -> pd.DataFrame:
    """Attach constant empirical ``cov_*`` columns to every row in ``frame``."""

    if source not in _SOURCE_COORDS:
        raise ValueError(f"unknown source {source!r}")
    if frame.empty or source not in covariance_payload:
        return frame
    covariance = np.asarray(covariance_payload[source]["covariance_m2"], dtype=float)
    dim = len(_SOURCE_COORDS[source])
    covariance = covariance.reshape((dim, dim))
    out = frame.copy()
    for i, j, suffix in _COVARIANCE_SUFFIXES[dim]:
        out[f"cov_{suffix}"] = float(covariance[i, j])
    out["uncertainty_model"] = "paper-empirical-covariance"
    return out


def _nearest_time_indices(reference_times_s: np.ndarray, query_times_s: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference_times_s, dtype=float).reshape(-1)
    query = np.asarray(query_times_s, dtype=float).reshape(-1)
    insertion = np.searchsorted(reference, query)
    right = np.clip(insertion, 0, reference.size - 1)
    left = np.clip(insertion - 1, 0, reference.size - 1)
    use_right = np.abs(reference[right] - query) < np.abs(reference[left] - query)
    return np.where(use_right, right, left)
