"""Restore strict validation for shared CV model-construction controls."""

from __future__ import annotations

import sys
from typing import Any

import numpy as np


def _finite_nonnegative_real_scalar(value: Any, *, name: str) -> float:
    """Return a finite non-negative real scalar or raise a stable error."""

    error = f"{name} must be a finite, non-negative real scalar"
    if isinstance(value, bool | np.bool_) or np.ma.is_masked(value):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    item = scalar.item()
    if isinstance(item, bool | np.bool_):
        raise ValueError(error)
    try:
        parsed = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(error)
    return parsed


def apply_kalman_model_control_validation_patch(kalman_module: Any) -> None:
    """Validate CV transition/process-noise controls across loaded import aliases."""

    if getattr(kalman_module, "_model_control_validation_patch_applied", False):
        return

    original_constant_velocity_matrix = kalman_module.constant_velocity_matrix
    original_white_acceleration_process_noise = kalman_module.white_acceleration_process_noise

    def constant_velocity_matrix(dt_s: float) -> np.ndarray:
        dt = _finite_nonnegative_real_scalar(dt_s, name="dt_s")
        return original_constant_velocity_matrix(dt)

    def white_acceleration_process_noise(
        dt_s: float,
        acceleration_std: float,
    ) -> np.ndarray:
        dt = _finite_nonnegative_real_scalar(dt_s, name="dt_s")
        std = _finite_nonnegative_real_scalar(
            acceleration_std,
            name="acceleration_std",
        )
        return original_white_acceleration_process_noise(dt, std)

    for module in tuple(sys.modules.values()):
        namespace = getattr(module, "__dict__", {}) if module is not None else {}
        if namespace.get("constant_velocity_matrix") is original_constant_velocity_matrix:
            setattr(module, "constant_velocity_matrix", constant_velocity_matrix)
        if (
            namespace.get("white_acceleration_process_noise")
            is original_white_acceleration_process_noise
        ):
            setattr(
                module,
                "white_acceleration_process_noise",
                white_acceleration_process_noise,
            )

    kalman_module.constant_velocity_matrix = constant_velocity_matrix
    kalman_module.white_acceleration_process_noise = white_acceleration_process_noise
    kalman_module._model_control_validation_patch_applied = True
