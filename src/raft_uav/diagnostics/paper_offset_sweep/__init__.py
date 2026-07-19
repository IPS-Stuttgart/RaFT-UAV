"""Compatibility hardening for paper offset-grid parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "paper_offset_sweep.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._paper_offset_sweep_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load paper-offset sweep implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_PARSE_GRID = _IMPL._parse_grid


def _parse_grid(spec: str) -> np.ndarray:
    """Parse a finite START,STOP,STEP offset grid."""

    raw_parts = str(spec).split(",")
    if len(raw_parts) != 3:
        raise ValueError("grid must have the form START,STOP,STEP")
    try:
        parts = np.asarray([float(part.strip()) for part in raw_parts], dtype=float)
    except ValueError as exc:
        raise ValueError("grid START, STOP, and STEP must be numeric") from exc
    if not bool(np.isfinite(parts).all()):
        raise ValueError("grid START, STOP, and STEP must be finite")
    return _LEGACY_PARSE_GRID(spec)


_IMPL._parse_grid = _parse_grid

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_parse_grid"] = _parse_grid

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
