"""Reject malformed controls before IMM transition-matrix construction."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_PATCH_MARKER = "_raft_uav_validates_finite_imm_transition_controls"


def _finite_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be a finite scalar")
    return parsed


def apply_imm_transition_validation_patch(module: ModuleType) -> None:
    """Patch the legacy IMM helpers with finite-control validation."""

    original_fixed_turn_rate_matrix = module.fixed_turn_rate_matrix
    original_uniform_ctmc_transition_matrix = module.uniform_ctmc_transition_matrix
    if getattr(original_fixed_turn_rate_matrix, _PATCH_MARKER, False) and getattr(
        original_uniform_ctmc_transition_matrix,
        _PATCH_MARKER,
        False,
    ):
        return

    @wraps(original_fixed_turn_rate_matrix)
    def fixed_turn_rate_matrix(dt_s: float, turn_rate_radps: float) -> np.ndarray:
        dt = _finite_float(dt_s, name="dt_s")
        turn_rate = _finite_float(turn_rate_radps, name="turn_rate_radps")
        return original_fixed_turn_rate_matrix(dt, turn_rate)

    @wraps(original_uniform_ctmc_transition_matrix)
    def uniform_ctmc_transition_matrix(
        n_modes: int,
        dt_s: float,
        mode_switch_time_constant_s: float,
    ) -> np.ndarray:
        dt = _finite_float(dt_s, name="dt_s")
        time_constant = _finite_float(
            mode_switch_time_constant_s,
            name="mode_switch_time_constant_s",
        )
        if time_constant <= 0.0:
            raise ValueError("mode_switch_time_constant_s must be positive and finite")
        return original_uniform_ctmc_transition_matrix(
            n_modes,
            dt,
            time_constant,
        )

    setattr(fixed_turn_rate_matrix, _PATCH_MARKER, True)
    setattr(uniform_ctmc_transition_matrix, _PATCH_MARKER, True)
    module.fixed_turn_rate_matrix = fixed_turn_rate_matrix
    module.uniform_ctmc_transition_matrix = uniform_ctmc_transition_matrix
