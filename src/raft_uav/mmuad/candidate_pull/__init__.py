"""Compatibility package validating MMUAD candidate-pull top-k controls.

The maintained implementation lives in the sibling ``candidate_pull.py``
module. This package preserves the public import path while rejecting malformed
``top_k`` values instead of silently truncating or clamping them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_pull.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pull_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate-pull implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TOPK_CANDIDATE_CENTERS = _IMPL.topk_candidate_centers
_ORIGINAL_CANDIDATE_CENTERS_FOR_RESULTS = _IMPL.candidate_centers_for_results


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without truncating malformed numeric values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not np.isfinite(number) or number <= 0.0 or not number.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def topk_candidate_centers(
    candidates: pd.DataFrame,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return candidate centers after validating the requested top-k count."""

    return _ORIGINAL_TOPK_CANDIDATE_CENTERS(
        candidates,
        top_k=_positive_integer(top_k, name="top_k"),
    )


def candidate_centers_for_results(
    candidates: pd.DataFrame,
    results: pd.DataFrame,
    current_xyz: np.ndarray,
    *,
    top_k: int = 5,
    time_tolerance_s: float = 0.5,
) -> pd.DataFrame:
    """Return row-wise centers after validating the requested top-k count."""

    return _ORIGINAL_CANDIDATE_CENTERS_FOR_RESULTS(
        candidates,
        results,
        current_xyz,
        top_k=_positive_integer(top_k, name="top_k"),
        time_tolerance_s=time_tolerance_s,
    )


_IMPL.topk_candidate_centers = topk_candidate_centers
_IMPL.candidate_centers_for_results = candidate_centers_for_results

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_positive_integer"] = _positive_integer
globals()["topk_candidate_centers"] = topk_candidate_centers
globals()["candidate_centers_for_results"] = candidate_centers_for_results

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
