"""Compatibility fix for no-op Track 5 acceleration repairs.

The maintained implementation lives in the sibling ``track5_acceleration_limit.py``
module. This package preserves the public import path while ensuring that a zero
repair blend is reported as a detected candidate, not as an applied trajectory
change.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_acceleration_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_acceleration_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 acceleration-limit implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_REPAIR_SEQUENCE = _IMPL._repair_sequence


class _Track5AccelerationLimitModule(ModuleType):
    """Module proxy that keeps runtime monkeypatches visible to legacy globals."""

    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if name == "_IMPL":
            return
        implementation = self.__dict__.get("_IMPL")
        if implementation is not None and hasattr(implementation, name):
            setattr(implementation, name, value)


def _repair_sequence(group, **kwargs):
    """Keep zero-blend runs diagnostic-only instead of marking rows as changed."""

    repaired, diagnostics = _ORIGINAL_REPAIR_SEQUENCE(group, **kwargs)
    if float(kwargs["repair_blend"]) != 0.0:
        return repaired, diagnostics

    repaired = repaired.copy()
    diagnostics = diagnostics.copy()
    repaired["acceleration_limit_applied"] = False
    repaired["acceleration_limit_iteration"] = 0
    repaired["acceleration_limit_displacement_m"] = 0.0
    diagnostics["acceleration_limit_applied"] = False
    diagnostics["acceleration_limit_iteration"] = 0
    diagnostics["acceleration_limit_displacement_m"] = 0.0
    return repaired, diagnostics


_IMPL._repair_sequence = _repair_sequence

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_repair_sequence"] = _repair_sequence
__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
sys.modules[__name__].__class__ = _Track5AccelerationLimitModule
