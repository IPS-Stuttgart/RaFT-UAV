"""Compatibility wrapper validating Track 5 Hampel integer controls.

The maintained implementation lives in the sibling ``track5_hampel_repair.py``
module. This package preserves the public import path while rejecting malformed
window radii and iteration counts before the implementation can silently coerce
them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_hampel_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_hampel_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Hampel-repair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_REPAIR = _IMPL.repair_track5_hampel_spikes
_ORIGINAL_REPAIR_SEQUENCE = _IMPL._repair_sequence
_ORIGINAL_REPAIR_XYZ_ONCE = _IMPL._repair_xyz_once


def _normalize_positive_integer(value: Any, *, field: str) -> int:
    """Return a positive integer scalar or raise a field-specific error."""

    message = f"{field} must be a positive finite integer"
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 1.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def repair_track5_hampel_spikes(submission, **kwargs):
    """Validate integer controls before running the legacy Hampel repair."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs.get("window_radius", 2),
        field="window_radius",
    )
    kwargs["iterations"] = _normalize_positive_integer(
        kwargs.get("iterations", 1),
        field="iterations",
    )
    return _ORIGINAL_REPAIR(submission, **kwargs)


def _repair_sequence(group, **kwargs):
    """Validate private sequence-loop integer controls for direct callers."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs["window_radius"],
        field="window_radius",
    )
    kwargs["iterations"] = _normalize_positive_integer(
        kwargs["iterations"],
        field="iterations",
    )
    return _ORIGINAL_REPAIR_SEQUENCE(group, **kwargs)


def _repair_xyz_once(work, xyz, original_xyz, **kwargs):
    """Validate the private local-window radius for direct callers."""

    kwargs["window_radius"] = _normalize_positive_integer(
        kwargs["window_radius"],
        field="window_radius",
    )
    return _ORIGINAL_REPAIR_XYZ_ONCE(work, xyz, original_xyz, **kwargs)


_IMPL.repair_track5_hampel_spikes = repair_track5_hampel_spikes
_IMPL._repair_sequence = _repair_sequence
_IMPL._repair_xyz_once = _repair_xyz_once

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["repair_track5_hampel_spikes"] = repair_track5_hampel_spikes
globals()["_repair_sequence"] = _repair_sequence
globals()["_repair_xyz_once"] = _repair_xyz_once
globals()["_normalize_positive_integer"] = _normalize_positive_integer

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
