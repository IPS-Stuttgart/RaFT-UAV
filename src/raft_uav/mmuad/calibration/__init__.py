"""Compatibility validation for MMUAD calibration transforms.

The maintained implementation lives in the sibling ``calibration.py`` module.
This package preserves the public import path while rejecting complex-valued
rotation and translation inputs before NumPy can silently discard their
imaginary components during real-valued coercion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "calibration.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._calibration_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_RIGID_TRANSFORM = _IMPL.RigidTransform


def _contains_complex(value: Any) -> bool:
    """Return whether an array-like value contains a complex scalar."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    try:
        values = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if np.iscomplexobj(values):
        return True
    if values.dtype != object:
        return False
    return any(_contains_complex(item) for item in values.flat)


def _reject_complex(value: Any, *, name: str) -> None:
    if _contains_complex(value):
        raise ValueError(f"{name} must contain real values")


class RigidTransform(_ORIGINAL_RIGID_TRANSFORM):
    """Rigid transform that rejects lossy complex-to-real coercion."""

    def __post_init__(self) -> None:
        _reject_complex(self.rotation, name="rotation")
        _reject_complex(self.translation_m, name="translation_m")
        super().__post_init__()


_IMPL.RigidTransform = RigidTransform

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_contains_complex"] = _contains_complex
globals()["_reject_complex"] = _reject_complex
globals()["RigidTransform"] = RigidTransform

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
