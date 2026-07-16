"""Compatibility fix for oracle-coverage candidate identifier normalization.

The maintained implementation lives in the sibling ``oracle_coverage.py``
module. This package preserves the public import path while preventing
fractional candidate identifiers from being silently truncated in diagnostics.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "oracle_coverage.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._oracle_coverage_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load oracle coverage implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _optional_int(value: object) -> int | None:
    """Return an integer-equivalent finite scalar without truncation."""

    number = _IMPL._optional_float(value)
    if number is None:
        return None
    rounded = np.rint(number)
    if number != rounded:
        return None
    return int(rounded)


_IMPL._optional_int = _optional_int

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_optional_int"] = _optional_int

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
