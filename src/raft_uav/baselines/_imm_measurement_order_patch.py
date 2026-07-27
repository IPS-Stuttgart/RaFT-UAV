"""Deterministic same-timestamp measurement ordering for the IMM baseline."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

from raft_uav.baselines.kalman import _tracking_measurement_order_key


def apply_imm_measurement_order_patch(module: ModuleType) -> None:
    """Make equal-time IMM updates independent of caller input order."""

    if getattr(module, "_measurement_order_patch_applied", False):
        return

    original = module.run_async_imm_baseline

    @wraps(original)
    def run_async_imm_baseline(
        measurements: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ordered = sorted(measurements, key=_tracking_measurement_order_key)
        return original(ordered, *args, **kwargs)

    module.run_async_imm_baseline = run_async_imm_baseline
    module._measurement_order_patch_applied = True
