"""Compatibility fix for radar-calibration measurement row filtering.

The maintained implementation lives in the sibling ``radar_calibration_audit.py``
module. This package preserves the public import path while dropping unusable
measurement rows before nearest-time matching.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_calibration_audit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._radar_calibration_audit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load radar calibration audit from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_LEGACY_FIT_YAW_OFFSET_ALTITUDE = _IMPL.fit_yaw_offset_altitude


def _empty_pairs():
    return _IMPL.MeasurementTruthPairs(
        measurement_times_s=np.array([], dtype=float),
        measurement_positions_m=np.empty((0, 3), dtype=float),
        truth_times_s=np.array([], dtype=float),
        truth_positions_m=np.empty((0, 3), dtype=float),
    )


def pair_measurements_to_truth(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    time_offset_s: float = 0.0,
    max_time_delta_s: float = 2.0,
):
    """Pair only finite measurement rows to nearest finite truth timestamps."""

    _IMPL._require_columns(
        measurements,
        ("time_s", *_IMPL.POSITION_COLUMNS),
        context="measurements",
    )
    _IMPL._require_columns(
        truth,
        ("time_s", *_IMPL.POSITION_COLUMNS),
        context="truth",
    )
    measurement_times = (
        measurements["time_s"].to_numpy(dtype=float) + float(time_offset_s)
    )
    measurement_positions = measurements.loc[
        :, _IMPL.POSITION_COLUMNS
    ].to_numpy(dtype=float)
    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth.loc[:, _IMPL.POSITION_COLUMNS].to_numpy(dtype=float)
    if truth_times.size == 0:
        raise ValueError("truth must not be empty")
    if measurement_times.size == 0:
        return _empty_pairs()

    usable_measurements = np.isfinite(measurement_times) & np.isfinite(
        measurement_positions
    ).all(axis=1)
    if not bool(usable_measurements.any()):
        return _empty_pairs()

    candidate_times = measurement_times[usable_measurements]
    candidate_positions = measurement_positions[usable_measurements]
    indices = _IMPL.nearest_time_indices(truth_times, candidate_times)
    delta_t = np.abs(truth_times[indices] - candidate_times)
    finite_matches = (
        np.isfinite(delta_t)
        & np.isfinite(truth_positions[indices]).all(axis=1)
        & (delta_t <= float(max_time_delta_s))
    )
    return _IMPL.MeasurementTruthPairs(
        measurement_times_s=candidate_times[finite_matches],
        measurement_positions_m=candidate_positions[finite_matches],
        truth_times_s=truth_times[indices][finite_matches],
        truth_positions_m=truth_positions[indices][finite_matches],
    )


def fit_yaw_offset_altitude(pairs):
    """Fit yaw only when the paired horizontal geometry constrains it."""

    _IMPL._require_pair_count(pairs, minimum=2)
    measured_xy = np.asarray(pairs.measurement_positions_m[:, :2], dtype=float)
    truth_xy = np.asarray(pairs.truth_positions_m[:, :2], dtype=float)
    measured_centered = measured_xy - np.mean(measured_xy, axis=0)
    truth_centered = truth_xy - np.mean(truth_xy, axis=0)
    covariance = measured_centered.T @ truth_centered

    # For a 2-D proper rotation R(theta), the Procrustes objective depends on
    # theta only through these cosine/sine coefficients. If both vanish, every
    # yaw has the same least-squares cost and the SVD's returned angle is an
    # arbitrary consequence of its basis choice rather than an observable fit.
    yaw_cos_information = float(covariance[0, 0] + covariance[1, 1])
    yaw_sin_information = float(covariance[0, 1] - covariance[1, 0])
    if float(np.hypot(yaw_cos_information, yaw_sin_information)) == 0.0:
        raise ValueError(
            "paired horizontal geometry does not constrain a unique yaw"
        )

    return _LEGACY_FIT_YAW_OFFSET_ALTITUDE(pairs)


_IMPL.pair_measurements_to_truth = pair_measurements_to_truth
_IMPL.fit_yaw_offset_altitude = fit_yaw_offset_altitude

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
