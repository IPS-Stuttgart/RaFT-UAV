"""Validate uncertainty controls for the core radar-association runner."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

from raft_uav.numeric import optional_float

_PATCH_MARKER = "_raft_uav_validates_core_radar_standard_deviations"


def _positive_finite_real(value: Any, *, name: str) -> float:
    """Return one finite, positive, non-Boolean real scalar."""

    parsed = optional_float(value)
    if parsed is None or parsed <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return parsed


def apply_radar_association_std_validation_patch(module: ModuleType) -> None:
    """Patch the core radar-association runner once."""

    original: Callable[..., Any] = module.run_async_cv_baseline_with_radar_association
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def validated_runner(
        *args: Any,
        radar_xy_std_m: Any = 25.0,
        radar_z_std_m: Any = 35.0,
        **kwargs: Any,
    ) -> Any:
        radar_xy_std = _positive_finite_real(
            radar_xy_std_m,
            name="radar_xy_std_m",
        )
        radar_z_std = _positive_finite_real(
            radar_z_std_m,
            name="radar_z_std_m",
        )
        return original(
            *args,
            radar_xy_std_m=radar_xy_std,
            radar_z_std_m=radar_z_std,
            **kwargs,
        )

    setattr(validated_runner, _PATCH_MARKER, True)
    module.run_async_cv_baseline_with_radar_association = validated_runner
