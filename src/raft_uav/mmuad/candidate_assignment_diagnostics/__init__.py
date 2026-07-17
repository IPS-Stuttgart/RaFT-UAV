"""Compatibility fix for exact candidate-rank parsing."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from raft_uav.numeric import optional_int as _safe_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load candidate-assignment diagnostics implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

# Candidate ranks are integer identifiers.  The legacy float round-trip silently
# truncated fractional values and lost precision for integers above 2**53.
_IMPL._safe_int = _safe_int

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_safe_int"] = _safe_int

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
