"""Compatibility package with strict branch-consensus control validation.

The maintained implementation lives in the sibling
``candidate_branch_consensus.py`` module. This package preserves the public
import path while rejecting malformed numeric controls before they can produce
non-finite consensus features or ranking scores.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_branch_consensus.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_branch_consensus_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load branch-consensus implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ATTACH_CANDIDATE_BRANCH_CONSENSUS = _IMPL.attach_candidate_branch_consensus


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


def _finite_nonnegative_scalar(value: Any, *, name: str) -> float:
    number = _finite_scalar(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return number


def _finite_positive_scalar(value: Any, *, name: str) -> float:
    number = _finite_scalar(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return number


def attach_candidate_branch_consensus(
    candidates,
    *,
    time_window_s: float = 0.05,
    time_scale_s: float | None = None,
    distance_gate_m: float = 5.0,
    distance_scale_m: float = 5.0,
    base_score_column: str = "ranker_score",
    score_output_column: str = _IMPL.DEFAULT_SCORE_OUTPUT_COLUMN,
    base_score_weight: float = 1.0,
    consensus_weight: float = 1.0,
    pair_advantage_weight: float = 0.25,
    branch_column: str | None = None,
    origin_column: str | None = None,
    exclude_same_origin_support: bool = True,
    replace_confidence: bool = False,
):
    """Attach branch consensus using finite, contract-preserving controls."""

    normalized_time_window = _finite_nonnegative_scalar(
        time_window_s,
        name="time_window_s",
    )
    normalized_time_scale = (
        None
        if time_scale_s is None
        else _finite_positive_scalar(time_scale_s, name="time_scale_s")
    )
    normalized_distance_gate = _finite_positive_scalar(
        distance_gate_m,
        name="distance_gate_m",
    )
    normalized_distance_scale = _finite_positive_scalar(
        distance_scale_m,
        name="distance_scale_m",
    )
    normalized_base_weight = _finite_scalar(
        base_score_weight,
        name="base_score_weight",
    )
    normalized_consensus_weight = _finite_scalar(
        consensus_weight,
        name="consensus_weight",
    )
    normalized_pair_weight = _finite_scalar(
        pair_advantage_weight,
        name="pair_advantage_weight",
    )
    return _ORIGINAL_ATTACH_CANDIDATE_BRANCH_CONSENSUS(
        candidates,
        time_window_s=normalized_time_window,
        time_scale_s=normalized_time_scale,
        distance_gate_m=normalized_distance_gate,
        distance_scale_m=normalized_distance_scale,
        base_score_column=base_score_column,
        score_output_column=score_output_column,
        base_score_weight=normalized_base_weight,
        consensus_weight=normalized_consensus_weight,
        pair_advantage_weight=normalized_pair_weight,
        branch_column=branch_column,
        origin_column=origin_column,
        exclude_same_origin_support=exclude_same_origin_support,
        replace_confidence=replace_confidence,
    )


_IMPL.attach_candidate_branch_consensus = attach_candidate_branch_consensus

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_scalar"] = _finite_scalar
globals()["_finite_nonnegative_scalar"] = _finite_nonnegative_scalar
globals()["_finite_positive_scalar"] = _finite_positive_scalar
globals()["attach_candidate_branch_consensus"] = attach_candidate_branch_consensus

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
