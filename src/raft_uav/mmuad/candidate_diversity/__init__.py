"""Compatibility wrapper for robust candidate-diversity inputs.

The maintained implementation lives in the sibling ``candidate_diversity.py``
module. This package preserves the public import path while coercing candidate
coordinates before spatial filtering so malformed rows are skipped rather than
raising during NumPy conversion. It also validates diversity controls before
legacy numeric coercion can silently clamp or truncate them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
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


def _finite_nonnegative_control(value: Any, *, name: str) -> float:
    """Return a finite nonnegative real scalar without Boolean coercion."""

    message = f"{name} must be a finite non-negative number"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(message)
    try:
        number = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(message)
    return number


def _positive_integer_control(value: Any, *, name: str) -> int:
    """Return a positive integer scalar without fractional truncation."""

    message = f"{name} must be a positive integer"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(message)
    try:
        number = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 1.0 or not number.is_integer():
        raise ValueError(message)
    return int(number)


def diversify_candidate_reservoir(rows, **kwargs):
    """Validate controls and coerce coordinates before diversity pruning."""

    normalized_kwargs = dict(kwargs)
    normalized_kwargs["radius_m"] = _finite_nonnegative_control(
        normalized_kwargs.get("radius_m", 1.0),
        name="radius_m",
    )
    normalized_kwargs["max_candidates_per_frame"] = _positive_integer_control(
        normalized_kwargs.get("max_candidates_per_frame", 40),
        name="max_candidates_per_frame",
    )

    frame = pd.DataFrame(rows).copy()
    for column in ("x_m", "y_m", "z_m"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return _ORIGINAL_DIVERSIFY(frame, **normalized_kwargs)


_IMPL._finite_nonnegative_control = _finite_nonnegative_control
_IMPL._positive_integer_control = _positive_integer_control
_IMPL.diversify_candidate_reservoir = diversify_candidate_reservoir

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_nonnegative_control"] = _finite_nonnegative_control
globals()["_positive_integer_control"] = _positive_integer_control
globals()["diversify_candidate_reservoir"] = diversify_candidate_reservoir

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
