"""Use strict absolute timestamp matching for IMM bootstrap suppression."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np


_imm = import_module("raft_uav.baselines.imm")
_ORIGINAL_IS_BOOTSTRAP_MEASUREMENT = (
    _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement
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


def install() -> None:
    """Install strict IMM bootstrap matching and provenance checks once."""

    if not getattr(_imm, "_imm_bootstrap_timestamp_patch_applied", False):
        _imm.AsyncInteractingMultipleModelTracker._is_bootstrap_measurement = (
            _is_bootstrap_measurement
        )
        _imm._imm_bootstrap_timestamp_patch_applied = True

    from raft_uav.baselines._bootstrap_measurement_provenance_patch import (
        install as install_provenance,
    )

    install_provenance()
