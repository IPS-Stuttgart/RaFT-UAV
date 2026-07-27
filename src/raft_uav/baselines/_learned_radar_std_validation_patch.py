"""Validate learned radar-association standard-deviation controls."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_PATCH_MARKER = "_raft_uav_validates_learned_radar_standard_deviations"


def _positive_finite_real(value: Any, *, name: str) -> float:
    """Return one finite, positive, unmasked, non-Boolean real scalar."""

    error = f"{name} must be a finite positive real scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    item = scalar.item()
    if np.ma.is_masked(item) or isinstance(item, (bool, np.bool_)):
        raise ValueError(error)
    try:
        parsed = float(item)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(error)
    return parsed


def _patch_runner(module: ModuleType) -> None:
    original: Callable[..., Any] = (
        module.run_async_cv_baseline_with_learned_radar_association
        if hasattr(module, "run_async_cv_baseline_with_learned_radar_association")
        else module.run_async_cv_baseline_with_stateful_learned_radar_association
    )
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated_runner(
        *args: Any,
        radar_xy_std_m: Any = 25.0,
        radar_z_std_m: Any = 35.0,
        **kwargs: Any,
    ) -> Any:
        radar_xy_std = _positive_finite_real(radar_xy_std_m, name="radar_xy_std_m")
        radar_z_std = _positive_finite_real(radar_z_std_m, name="radar_z_std_m")
        return original(
            *args,
            radar_xy_std_m=radar_xy_std,
            radar_z_std_m=radar_z_std,
            **kwargs,
        )

    setattr(validated_runner, _PATCH_MARKER, True)
    if hasattr(module, "run_async_cv_baseline_with_learned_radar_association"):
        module.run_async_cv_baseline_with_learned_radar_association = validated_runner
    else:
        module.run_async_cv_baseline_with_stateful_learned_radar_association = validated_runner


def apply_learned_radar_std_validation_patch(
    per_frame_module: ModuleType,
    stateful_module: ModuleType,
) -> None:
    """Patch both learned radar-association runners with strict std validation."""

    _patch_runner(per_frame_module)
    _patch_runner(stateful_module)
