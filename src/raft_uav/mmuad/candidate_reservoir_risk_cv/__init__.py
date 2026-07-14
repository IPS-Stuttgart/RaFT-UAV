"""Compatibility wrapper normalizing direct risk-CV candidate inputs.

The maintained implementation lives in the sibling
``candidate_reservoir_risk_cv.py`` module. This package preserves the public
import path while applying the same candidate schema normalization used by the
CLI before sequence-fold discovery.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

from raft_uav.mmuad.schema import normalize_candidate_columns

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_risk_cv.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_risk_cv_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load risk-aware reservoir CV implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_VALIDATED_INPUTS = _IMPL._validated_inputs


def _validated_inputs_with_candidate_normalization(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Normalize candidate aliases and identifiers before fold discovery."""

    candidate_rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    return _ORIGINAL_VALIDATED_INPUTS(candidate_rows, truth)


_IMPL._validated_inputs = _validated_inputs_with_candidate_normalization

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
