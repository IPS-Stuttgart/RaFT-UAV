"""Compatibility package validating candidate-consensus quota controls.

The maintained implementation lives in the sibling
``candidate_consensus_quota.py`` module. This package preserves the public import
path while validating scalar controls before empty-input returns and before
numeric conversion can leak implementation-specific exceptions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_consensus_quota.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_consensus_quota_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate consensus quota from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

CandidateFrame = _IMPL.CandidateFrame
ReservoirConfig = _IMPL.ReservoirConfig
_ORIGINAL_BUILD = _IMPL.build_consensus_quota_reservoir


def _finite_scalar(value: Any, *, name: str) -> float:
    """Return one finite non-Boolean scalar with a stable validation error."""

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


def _integer_control(
    value: Any,
    *,
    name: str,
    positive: bool,
) -> int:
    """Return an integer-valued scalar respecting the requested lower bound."""

    message = (
        f"{name} must be a positive integer"
        if positive
        else f"{name} must be a non-negative integer"
    )
    try:
        number = _finite_scalar(value, name=name)
    except ValueError as exc:
        raise ValueError(message) from exc
    lower_bound = 1 if positive else 0
    if not number.is_integer() or number < lower_bound:
        raise ValueError(message)
    return int(number)


def _normalize_controls(
    *,
    consensus_top_n: Any,
    min_neighbor_count: Any,
    min_unique_source_count: Any,
    max_nearest_distance_m: Any,
    max_per_origin: Any,
    max_per_source: Any,
    time_window_s: Any,
    time_scale_s: Any | None,
    distance_gate_m: Any,
    distance_scale_m: Any,
) -> dict[str, int | float | None]:
    """Validate and normalize every non-weight consensus quota control."""

    normalized: dict[str, int | float | None] = {
        "consensus_top_n": _integer_control(
            consensus_top_n,
            name="consensus_top_n",
            positive=False,
        ),
        "min_neighbor_count": _integer_control(
            min_neighbor_count,
            name="min_neighbor_count",
            positive=False,
        ),
        "min_unique_source_count": _integer_control(
            min_unique_source_count,
            name="min_unique_source_count",
            positive=False,
        ),
        "max_per_origin": _integer_control(
            max_per_origin,
            name="max_per_origin",
            positive=True,
        ),
        "max_per_source": _integer_control(
            max_per_source,
            name="max_per_source",
            positive=False,
        ),
    }
    for name, value in (
        ("max_nearest_distance_m", max_nearest_distance_m),
        ("distance_gate_m", distance_gate_m),
        ("distance_scale_m", distance_scale_m),
    ):
        number = _finite_scalar(value, name=name)
        if number <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        normalized[name] = number

    window = _finite_scalar(time_window_s, name="time_window_s")
    if window < 0.0:
        raise ValueError("time_window_s must be non-negative and finite")
    normalized["time_window_s"] = window

    if time_scale_s is None:
        normalized["time_scale_s"] = None
    else:
        scale = _finite_scalar(time_scale_s, name="time_scale_s")
        if scale <= 0.0:
            raise ValueError("time_scale_s must be positive and finite when provided")
        normalized["time_scale_s"] = scale
    return normalized


def _validate_controls(
    *,
    consensus_top_n: Any,
    min_neighbor_count: Any,
    min_unique_source_count: Any,
    max_nearest_distance_m: Any,
    max_per_origin: Any,
    max_per_source: Any,
    time_window_s: Any,
    time_scale_s: Any | None,
    distance_gate_m: Any,
    distance_scale_m: Any,
) -> None:
    """Validate controls using the legacy helper's public calling convention."""

    _normalize_controls(
        consensus_top_n=consensus_top_n,
        min_neighbor_count=min_neighbor_count,
        min_unique_source_count=min_unique_source_count,
        max_nearest_distance_m=max_nearest_distance_m,
        max_per_origin=max_per_origin,
        max_per_source=max_per_source,
        time_window_s=time_window_s,
        time_scale_s=time_scale_s,
        distance_gate_m=distance_gate_m,
        distance_scale_m=distance_scale_m,
    )


def build_consensus_quota_reservoir(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    reservoir_config: ReservoirConfig | None = None,
    consensus_top_n: int = 2,
    min_neighbor_count: int = 1,
    min_unique_source_count: int = 1,
    max_nearest_distance_m: float = 5.0,
    max_per_origin: int = 1,
    max_per_source: int = 0,
    selection_score_column: str = "branch_consensus_score",
    time_window_s: float = 0.05,
    time_scale_s: float | None = None,
    distance_gate_m: float = 5.0,
    distance_scale_m: float = 5.0,
    base_score_column: str = "ranker_score",
    base_score_weight: float = 1.0,
    consensus_weight: float = 1.0,
    pair_advantage_weight: float = 0.25,
    branch_column: str | None = None,
    origin_column: str | None = None,
    exclude_same_origin_support: bool = True,
) -> CandidateFrame:
    """Build a consensus quota reservoir after normalizing every scalar control."""

    controls = _normalize_controls(
        consensus_top_n=consensus_top_n,
        min_neighbor_count=min_neighbor_count,
        min_unique_source_count=min_unique_source_count,
        max_nearest_distance_m=max_nearest_distance_m,
        max_per_origin=max_per_origin,
        max_per_source=max_per_source,
        time_window_s=time_window_s,
        time_scale_s=time_scale_s,
        distance_gate_m=distance_gate_m,
        distance_scale_m=distance_scale_m,
    )
    weights = {
        name: _finite_scalar(value, name=name)
        for name, value in (
            ("base_score_weight", base_score_weight),
            ("consensus_weight", consensus_weight),
            ("pair_advantage_weight", pair_advantage_weight),
        )
    }
    return _ORIGINAL_BUILD(
        candidates,
        reservoir_config=reservoir_config,
        consensus_top_n=controls["consensus_top_n"],
        min_neighbor_count=controls["min_neighbor_count"],
        min_unique_source_count=controls["min_unique_source_count"],
        max_nearest_distance_m=controls["max_nearest_distance_m"],
        max_per_origin=controls["max_per_origin"],
        max_per_source=controls["max_per_source"],
        selection_score_column=selection_score_column,
        time_window_s=controls["time_window_s"],
        time_scale_s=controls["time_scale_s"],
        distance_gate_m=controls["distance_gate_m"],
        distance_scale_m=controls["distance_scale_m"],
        base_score_column=base_score_column,
        base_score_weight=weights["base_score_weight"],
        consensus_weight=weights["consensus_weight"],
        pair_advantage_weight=weights["pair_advantage_weight"],
        branch_column=branch_column,
        origin_column=origin_column,
        exclude_same_origin_support=exclude_same_origin_support,
    )


_IMPL._finite_scalar = _finite_scalar
_IMPL._integer_control = _integer_control
_IMPL._normalize_controls = _normalize_controls
_IMPL._validate_controls = _validate_controls
_IMPL.build_consensus_quota_reservoir = build_consensus_quota_reservoir

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_finite_scalar"] = _finite_scalar
globals()["_integer_control"] = _integer_control
globals()["_normalize_controls"] = _normalize_controls
globals()["_validate_controls"] = _validate_controls
globals()["build_consensus_quota_reservoir"] = build_consensus_quota_reservoir

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
