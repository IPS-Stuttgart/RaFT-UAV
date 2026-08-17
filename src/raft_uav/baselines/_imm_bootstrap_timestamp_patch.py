"""Harden IMM bootstrap timing and preserve scalar validation contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from raft_uav.baselines._kalman_timestamp_validation_patch import (
    _finite_timestamp_seconds,
)


_imm = import_module("raft_uav.baselines.imm")
_ORIGINAL_IMM_TRACKER_INIT = _imm.AsyncInteractingMultipleModelTracker.__init__
_ORIGINAL_IS_BOOTSTRAP_MEASUREMENT = (
    _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement
)
_ORIGINAL_COAST_TO = _imm.AsyncInteractingMultipleModelTracker.coast_to
_ORIGINAL_UNIFORM_CTMC_TRANSITION_MATRIX = _imm.uniform_ctmc_transition_matrix


def _finite_scalar_seconds(value: Any, *, field_name: str) -> float:
    """Validate a scalar time quantity without calling it a timestamp."""

    try:
        return _finite_timestamp_seconds(value, field_name=field_name)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a finite scalar") from exc


def _imm_tracker_init(
    self: Any,
    initial_position: np.ndarray,
    initial_time_s: float,
    initial_position_std_m: float = 50.0,
    initial_velocity_std_mps: float = 15.0,
    acceleration_std_mps2: float = 4.0,
    modes: Any = None,
    initial_mode_probabilities: Any = None,
    mode_switch_time_constant_s: float = 20.0,
) -> None:
    """Preserve scalar diagnostics for the IMM mode-switch time constant."""

    validated_time_constant_s = _finite_scalar_seconds(
        mode_switch_time_constant_s,
        field_name="mode_switch_time_constant_s",
    )
    _ORIGINAL_IMM_TRACKER_INIT(
        self,
        initial_position,
        initial_time_s,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        acceleration_std_mps2=acceleration_std_mps2,
        modes=modes,
        initial_mode_probabilities=initial_mode_probabilities,
        mode_switch_time_constant_s=validated_time_constant_s,
    )


def _is_bootstrap_measurement(self: Any, measurement: Any) -> bool:
    """Reject later measurements hidden by NumPy's default relative tolerance."""

    if not self._initial_update_pending:
        return False
    if not np.isclose(
        float(measurement.time_s),
        float(self.current_time_s),
        rtol=0.0,
        atol=1.0e-9,
    ):
        return False
    return bool(_ORIGINAL_IS_BOOTSTRAP_MEASUREMENT(self, measurement))


def _coast_to(self: Any, time_s: float) -> None:
    """Validate time before IMM coast bookkeeping consumes bootstrap state."""

    validated_time_s = _finite_timestamp_seconds(time_s, field_name="time_s")
    if validated_time_s < float(self.current_time_s) - 1.0e-9:
        raise ValueError("measurements must be processed in chronological order")
    _ORIGINAL_COAST_TO(self, validated_time_s)


def _uniform_ctmc_transition_matrix(
    n_modes: int,
    dt_s: float,
    mode_switch_time_constant_s: float,
) -> np.ndarray:
    """Keep transition intervals on the established finite-scalar contract."""

    validated_dt_s = _finite_scalar_seconds(dt_s, field_name="dt_s")
    validated_time_constant_s = _finite_scalar_seconds(
        mode_switch_time_constant_s,
        field_name="mode_switch_time_constant_s",
    )
    return _ORIGINAL_UNIFORM_CTMC_TRANSITION_MATRIX(
        n_modes,
        dt_s=validated_dt_s,
        mode_switch_time_constant_s=validated_time_constant_s,
    )


def install() -> None:
    """Install strict IMM bootstrap timing and scalar validation once."""

    if not getattr(_imm, "_imm_bootstrap_timestamp_patch_applied", False):
        _imm.AsyncInteractingMultipleModelTracker.__init__ = _imm_tracker_init
        _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement = (
            _is_bootstrap_measurement
        )
        _imm.AsyncInteractingMultipleModelTracker.coast_to = _coast_to
        _imm.uniform_ctmc_transition_matrix = _uniform_ctmc_transition_matrix
        _imm._imm_bootstrap_timestamp_patch_applied = True

    from raft_uav.baselines._bootstrap_measurement_provenance_patch import (
        install as install_provenance,
    )

    install_provenance()
