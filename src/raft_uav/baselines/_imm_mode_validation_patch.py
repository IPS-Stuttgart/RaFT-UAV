"""Validate IMM mode definitions before tracker construction and prediction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_raft_uav_validates_imm_mode_definitions"


def _finite_real_scalar(
    value: Any,
    *,
    name: str,
    nonnegative: bool = False,
) -> float:
    """Return one finite real scalar, optionally constrained to be non-negative."""

    qualifier = "finite nonnegative real scalar" if nonnegative else "finite real scalar"
    error = f"{name} must be a {qualifier}"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    item = scalar.item()
    if (
        np.ma.is_masked(item)
        or isinstance(item, (bool, np.bool_))
        or np.iscomplexobj(item)
    ):
        raise ValueError(error)
    try:
        parsed = float(item)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(parsed) or (nonnegative and parsed < 0.0):
        raise ValueError(error)
    return parsed


def _mode_name(value: Any) -> str:
    """Return a non-empty string suitable for diagnostic probability-map keys."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("IMM mode name must be a non-empty string")
    return value


def apply_imm_mode_validation_patch(module: ModuleType) -> None:
    """Install validated IMM mode construction and unique-name enforcement."""

    original_mode_class = module.IMMMode
    original_tracker_init = module.AsyncInteractingMultipleModelTracker.__init__
    if getattr(original_mode_class, _PATCH_MARKER, False) and getattr(
        original_tracker_init,
        _PATCH_MARKER,
        False,
    ):
        return

    @dataclass(frozen=True)
    class IMMMode(original_mode_class):
        """Validated replacement preserving the public IMM mode interface."""

        def __post_init__(self) -> None:
            object.__setattr__(self, "name", _mode_name(self.name))
            object.__setattr__(
                self,
                "acceleration_std_mps2",
                _finite_real_scalar(
                    self.acceleration_std_mps2,
                    name="acceleration_std_mps2",
                    nonnegative=True,
                ),
            )
            object.__setattr__(
                self,
                "turn_rate_radps",
                _finite_real_scalar(
                    self.turn_rate_radps,
                    name="turn_rate_radps",
                ),
            )

    IMMMode.__module__ = module.__name__
    IMMMode.__qualname__ = "IMMMode"
    setattr(IMMMode, _PATCH_MARKER, True)

    @wraps(original_tracker_init)
    def validated_tracker_init(
        self: object,
        initial_position: np.ndarray,
        initial_time_s: float,
        initial_position_std_m: float = 50.0,
        initial_velocity_std_mps: float = 15.0,
        acceleration_std_mps2: float = 4.0,
        modes: Any = None,
        initial_mode_probabilities: Any = None,
        mode_switch_time_constant_s: float = 20.0,
    ) -> None:
        selected_modes = None if modes is None else tuple(modes)
        if selected_modes is not None:
            if any(not isinstance(mode, original_mode_class) for mode in selected_modes):
                raise ValueError("modes must contain only IMMMode instances")
            names = tuple(mode.name for mode in selected_modes)
            if len(names) != len(set(names)):
                raise ValueError("IMM mode names must be unique")
        original_tracker_init(
            self,
            initial_position,
            initial_time_s,
            initial_position_std_m,
            initial_velocity_std_mps,
            acceleration_std_mps2,
            selected_modes,
            initial_mode_probabilities,
            mode_switch_time_constant_s,
        )

    setattr(validated_tracker_init, _PATCH_MARKER, True)
    module.IMMMode = IMMMode
    module.AsyncInteractingMultipleModelTracker.__init__ = validated_tracker_init
