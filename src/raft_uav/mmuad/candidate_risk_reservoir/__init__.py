"""Compatibility wrapper with finite candidate-risk control validation.

The maintained implementation lives in the sibling
``candidate_risk_reservoir.py`` module. This package preserves the public import
path while rejecting non-finite uncertainty controls before they can corrupt
risk scores and reservoir selection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_risk_reservoir.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_risk_reservoir_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load candidate risk reservoir from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ATTACH_CANDIDATE_RISK_SCORE = _IMPL.attach_candidate_risk_score


def _finite_float(value: Any, *, name: str) -> float:
    """Return a finite floating-point control with a field-specific error."""

    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _nonnegative_float(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_float(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def attach_candidate_risk_score(
    candidates,
    *,
    score_column: str = "candidate_class_calibrated_score",
    fallback_score_column: str = "ranker_score",
    sigma_column: str = "predicted_sigma_m",
    output_score_column: str = _IMPL.DEFAULT_OUTPUT_SCORE_COLUMN,
    mode: str = "logit-minus-log-sigma",
    uncertainty_weight: float = 1.0,
    sigma_floor_m: float = 1.0,
):
    """Attach risk scores after validating finite tuning controls."""

    normalized_weight = _nonnegative_float(
        uncertainty_weight,
        name="uncertainty_weight",
    )
    normalized_floor = _positive_float(
        sigma_floor_m,
        name="sigma_floor_m",
    )
    return _ORIGINAL_ATTACH_CANDIDATE_RISK_SCORE(
        candidates,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        sigma_column=sigma_column,
        output_score_column=output_score_column,
        mode=mode,
        uncertainty_weight=normalized_weight,
        sigma_floor_m=normalized_floor,
    )


_IMPL.attach_candidate_risk_score = attach_candidate_risk_score

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_float"] = _finite_float
globals()["_nonnegative_float"] = _nonnegative_float
globals()["_positive_float"] = _positive_float
globals()["attach_candidate_risk_score"] = attach_candidate_risk_score

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
