"""Use strict absolute timestamp matching for IMM bootstrap suppression."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np

from raft_uav.baselines._kalman_timestamp_validation_patch import (
    _finite_timestamp_seconds,
)


_imm = import_module("raft_uav.baselines.imm")
_ORIGINAL_IS_BOOTSTRAP_MEASUREMENT = (
    _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement
)
_ORIGINAL_COAST_TO = _imm.AsyncInteractingMultipleModelTracker.coast_to


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


def install() -> None:
    """Install strict IMM bootstrap matching and provenance checks once."""

    if not getattr(_imm, "_imm_bootstrap_timestamp_patch_applied", False):
        _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement = (
            _is_bootstrap_measurement
        )
        _imm.AsyncInteractingMultipleModelTracker.coast_to = _coast_to
        _imm._imm_bootstrap_timestamp_patch_applied = True

    from raft_uav.baselines._bootstrap_measurement_provenance_patch import (
        install as install_provenance,
    )

    install_provenance()
