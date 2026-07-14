"""Compatibility package validating candidate-assignment action-plan weights.

The maintained implementation lives in the sibling
``candidate_assignment_action_plan.py`` module. This package preserves the
public import path while rejecting non-finite ranking controls before they can
produce unusable priority scores or appear valid on empty diagnostic inputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_action_plan.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_action_plan_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate assignment action plan from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_CANDIDATE_ASSIGNMENT_ACTION_PLAN = (
    _IMPL.build_candidate_assignment_action_plan
)


def _finite_scalar(value: Any, *, name: str) -> float:
    """Return a finite non-Boolean scalar with a field-specific error."""

    message = f"{name} must be a finite scalar"
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def build_candidate_assignment_action_plan(
    blocks,
    *,
    top_n_blocks: int = 20,
    duration_weight: float = 1.0,
    frame_weight: float = 1.0,
    error_weight: float = 1.0,
    regret_weight: float = 1.0,
    buried_weight: float = 0.5,
):
    """Build an action plan after validating every ranking weight."""

    normalized_weights = {
        name: _finite_scalar(value, name=name)
        for name, value in (
            ("duration_weight", duration_weight),
            ("frame_weight", frame_weight),
            ("error_weight", error_weight),
            ("regret_weight", regret_weight),
            ("buried_weight", buried_weight),
        )
    }
    return _ORIGINAL_BUILD_CANDIDATE_ASSIGNMENT_ACTION_PLAN(
        blocks,
        top_n_blocks=top_n_blocks,
        **normalized_weights,
    )


_IMPL._finite_scalar = _finite_scalar
_IMPL.build_candidate_assignment_action_plan = (
    build_candidate_assignment_action_plan
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_scalar"] = _finite_scalar
globals()["build_candidate_assignment_action_plan"] = (
    build_candidate_assignment_action_plan
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
