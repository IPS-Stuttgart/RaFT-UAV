"""Compatibility wrapper for robust candidate-diversity numeric inputs.

The maintained implementation lives in the sibling ``candidate_diversity.py``
module. This package preserves the public import path while coercing candidate
coordinates before spatial filtering so malformed rows are skipped rather than
raising during NumPy conversion.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_diversity.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_diversity_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate-diversity implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_DIVERSIFY = _IMPL.diversify_candidate_reservoir


def diversify_candidate_reservoir(rows, **kwargs):
    """Coerce coordinate columns before delegating to diversity pruning."""

    frame = pd.DataFrame(rows).copy()
    for column in ("x_m", "y_m", "z_m"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _ORIGINAL_DIVERSIFY(frame, **kwargs)


_IMPL.diversify_candidate_reservoir = diversify_candidate_reservoir

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["diversify_candidate_reservoir"] = diversify_candidate_reservoir

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
