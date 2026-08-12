"""Runtime validation for radar measurement-conversion controls."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float


_aerpaw = import_module("raft_uav.io.aerpaw")
_ORIGINAL_RADAR_MEASUREMENTS_TO_ENU = _aerpaw.radar_measurements_to_enu


def _validated_positive_std(value: object, *, name: str) -> float:
    number = optional_float(value)
    if number is None or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return number


def _validated_boolean(value: object, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    raise ValueError(f"{name} must be boolean, got {value!r}")


@wraps(_ORIGINAL_RADAR_MEASUREMENTS_TO_ENU)
def _radar_measurements_to_enu(
    radar: pd.DataFrame,
    projector: Any = None,
    truth_origin_time: pd.Timestamp | None = None,
    default_xy_std_m: float = 25.0,
    default_z_std_m: float = 35.0,
    default_velocity_std_mps: float = 12.0,
    include_velocity: bool = False,
    clock_offset_s: float = _aerpaw.DEFAULT_RADAR_CLOCK_OFFSET_S,
) -> list[Any]:
    """Validate controls and enforce explicit 6-D velocity measurements."""

    validated_include_velocity = _validated_boolean(
        include_velocity,
        name="include_velocity",
    )
    validated_xy_std = _validated_positive_std(
        default_xy_std_m,
        name="default_xy_std_m",
    )
    validated_z_std = _validated_positive_std(
        default_z_std_m,
        name="default_z_std_m",
    )
    validated_velocity_std = default_velocity_std_mps
    if validated_include_velocity:
        validated_velocity_std = _validated_positive_std(
            default_velocity_std_mps,
            name="default_velocity_std_mps",
        )

    measurements = _ORIGINAL_RADAR_MEASUREMENTS_TO_ENU(
        radar,
        projector=projector,
        truth_origin_time=truth_origin_time,
        default_xy_std_m=validated_xy_std,
        default_z_std_m=validated_z_std,
        default_velocity_std_mps=validated_velocity_std,
        include_velocity=validated_include_velocity,
        clock_offset_s=clock_offset_s,
    )
    if validated_include_velocity:
        invalid_rows = [
            index
            for index, measurement in enumerate(measurements)
            if np.asarray(measurement.vector).reshape(-1).size != 6
        ]
        if invalid_rows:
            preview = ", ".join(str(index) for index in invalid_rows[:8])
            if len(invalid_rows) > 8:
                preview = f"{preview}, ..."
            raise ValueError(
                "include_velocity=True requires finite east, north, and down "
                "radar velocity components for every row; invalid row positions "
                f"[{preview}]"
            )
    return measurements


def apply_radar_measurement_validation_patch() -> None:
    """Install radar measurement-control validation once per interpreter."""

    if getattr(_aerpaw, "_radar_measurement_validation_patch_applied", False):
        return
    _aerpaw.radar_measurements_to_enu = _radar_measurements_to_enu
    _aerpaw._radar_measurement_validation_patch_applied = True
