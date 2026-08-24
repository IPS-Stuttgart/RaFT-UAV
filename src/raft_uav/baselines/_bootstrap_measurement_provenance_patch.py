"""Require measurement provenance before suppressing an initial tracker update."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


_kalman = import_module("raft_uav.baselines.kalman")
_imm = import_module("raft_uav.baselines.imm")
_ORIGINAL_KALMAN_INIT = _kalman.AsyncConstantVelocityKalmanTracker.__init__
_ORIGINAL_KALMAN_IS_BOOTSTRAP = (
    _kalman.AsyncConstantVelocityKalmanTracker._is_bootstrap_measurement
)
_ORIGINAL_IMM_INIT = _imm.AsyncInteractingMultipleModelTracker.__init__
_ORIGINAL_IMM_IS_BOOTSTRAP = (
    _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement
)
_BOOTSTRAP_VECTOR_ATTR = "_bootstrap_measurement_vector_source"


def _kalman_init(
    self: Any,
    initial_position: np.ndarray,
    initial_time_s: float,
    initial_position_std_m: float = 50.0,
    initial_velocity_std_mps: float = 15.0,
    acceleration_std_mps2: float = 4.0,
) -> None:
    """Remember the exact input object used to seed the CV tracker."""

    _ORIGINAL_KALMAN_INIT(
        self,
        initial_position,
        initial_time_s,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        acceleration_std_mps2=acceleration_std_mps2,
    )
    setattr(self, _BOOTSTRAP_VECTOR_ATTR, initial_position)


def _imm_init(
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
    """Remember the exact input object used to seed the IMM tracker."""

    _ORIGINAL_IMM_INIT(
        self,
        initial_position,
        initial_time_s,
        initial_position_std_m=initial_position_std_m,
        initial_velocity_std_mps=initial_velocity_std_mps,
        acceleration_std_mps2=acceleration_std_mps2,
        modes=modes,
        initial_mode_probabilities=initial_mode_probabilities,
        mode_switch_time_constant_s=mode_switch_time_constant_s,
    )
    setattr(self, _BOOTSTRAP_VECTOR_ATTR, initial_position)


def _has_bootstrap_provenance(self: Any, measurement: Any) -> bool:
    """Return whether ``measurement`` is the object that seeded the tracker."""

    if not hasattr(self, _BOOTSTRAP_VECTOR_ATTR):
        return False
    return getattr(measurement, "vector", None) is getattr(
        self,
        _BOOTSTRAP_VECTOR_ATTR,
    )


def _kalman_is_bootstrap_measurement(self: Any, measurement: Any) -> bool:
    """Suppress only the measurement object that supplied the initial state."""

    if not _has_bootstrap_provenance(self, measurement):
        return False
    return bool(_ORIGINAL_KALMAN_IS_BOOTSTRAP(self, measurement))


def _imm_is_bootstrap_measurement(self: Any, measurement: Any) -> bool:
    """Suppress only the measurement object that supplied the initial state."""

    if not _has_bootstrap_provenance(self, measurement):
        return False
    return bool(_ORIGINAL_IMM_IS_BOOTSTRAP(self, measurement))


def install() -> None:
    """Install provenance-aware bootstrap suppression once."""

    if not getattr(_kalman, "_bootstrap_measurement_provenance_patch_applied", False):
        _kalman.AsyncConstantVelocityKalmanTracker.__init__ = _kalman_init
        _kalman.AsyncConstantVelocityKalmanTracker._is_bootstrap_measurement = (
            _kalman_is_bootstrap_measurement
        )
        _kalman._bootstrap_measurement_provenance_patch_applied = True

    if not getattr(_imm, "_bootstrap_measurement_provenance_patch_applied", False):
        _imm.AsyncInteractingMultipleModelTracker.__init__ = _imm_init
        _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement = (
            _imm_is_bootstrap_measurement
        )
        _imm._bootstrap_measurement_provenance_patch_applied = True
