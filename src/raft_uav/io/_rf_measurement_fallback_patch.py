"""Runtime fix for normalized RF measurement covariance fallbacks."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float


_aerpaw = import_module("raft_uav.io.aerpaw")
_ORIGINAL_RF_MEASUREMENTS_TO_ENU = _aerpaw.rf_measurements_to_enu


def _positive_std(value: object) -> float | None:
    number = optional_float(value)
    return number if number is not None and number > 0.0 else None


def _sanitized_std_values(frame: pd.DataFrame, default_std: float) -> np.ndarray:
    if "std_m" not in frame.columns:
        return np.full(len(frame), default_std, dtype=float)

    values: list[float] = []
    for value in frame["std_m"].to_numpy(dtype=object):
        parsed = _positive_std(value)
        values.append(default_std if parsed is None else parsed)
    return np.asarray(values, dtype=float)


@wraps(_ORIGINAL_RF_MEASUREMENTS_TO_ENU)
def _rf_measurements_to_enu(
    rf: pd.DataFrame,
    projector: Any = None,
    truth_origin_time: pd.Timestamp | None = None,
    default_std_m: float = 75.0,
    clock_offset_s: float = _aerpaw.DEFAULT_RF_CLOCK_OFFSET_S,
) -> list[Any]:
    """Apply the documented default covariance to normalized RF rows safely."""

    default_std = _positive_std(default_std_m)
    if default_std is None:
        raise ValueError(
            f"default_std_m must be finite and positive, got {default_std_m!r}"
        )

    frame = rf
    if "east_m" in frame.columns:
        frame = frame.copy()
        frame["std_m"] = _sanitized_std_values(frame, default_std)

    return _ORIGINAL_RF_MEASUREMENTS_TO_ENU(
        frame,
        projector=projector,
        truth_origin_time=truth_origin_time,
        default_std_m=default_std,
        clock_offset_s=clock_offset_s,
    )


def apply_rf_measurement_fallback_patch() -> None:
    """Install normalized-RF fallback handling once per interpreter."""

    if getattr(_aerpaw, "_rf_measurement_fallback_patch_applied", False):
        return
    _aerpaw.rf_measurements_to_enu = _rf_measurements_to_enu
    _aerpaw._rf_measurement_fallback_patch_applied = True
