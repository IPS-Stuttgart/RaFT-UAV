"""Compatibility fixes for spatial-diversity reservoir selection.

The maintained implementation lives in the sibling
``candidate_reservoir_spatial.py`` module. This package preserves the public
import path while validating cap controls and spatial scales exactly, and while
treating malformed ranking scores as missing before candidate selection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import _finite_numeric_column
from raft_uav.numeric import optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_spatial.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_spatial_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load spatial reservoir implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_SPATIAL_DIVERSITY_CAP_RESERVOIR = _IMPL.spatial_diversity_cap_reservoir


def _nonnegative_integer(value: object, *, name: str) -> int:
    """Return an exact non-negative integer control."""

    parsed = optional_int(value)
    if parsed is None or parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed


def _positive_finite_scale(value: Any, *, name: str) -> float:
    """Return a finite positive non-Boolean real scalar scale."""

    message = f"{name} must be a finite positive real scalar"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or scalar.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        scale = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(message)
    return scale


def _score(
    rows: pd.DataFrame,
    score_column: str,
    fallback_score_column: str,
) -> pd.Series:
    """Return finite real ranking scores with documented fallback semantics."""

    primary = _finite_numeric_column(rows, score_column, default=np.nan)
    fallback = _finite_numeric_column(rows, fallback_score_column, default=1.0)
    return primary.fillna(fallback).fillna(0.0).astype(float)


def spatial_diversity_cap_reservoir(
    reservoir: pd.DataFrame,
    *,
    max_candidates_per_frame: object = 40,
    min_per_source: object = 1,
    min_per_branch: object = 1,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "confidence",
    branch_column: str = "candidate_branch",
    spatial_diversity_weight: float = 1.0,
    spatial_diversity_scale_m: float = 10.0,
    spatial_distance_cap_m: float = 50.0,
) -> pd.DataFrame:
    """Cap candidates after validating controls and ranking scores."""

    cap = _nonnegative_integer(
        max_candidates_per_frame,
        name="max_candidates_per_frame",
    )
    source_quota = _nonnegative_integer(min_per_source, name="min_per_source")
    branch_quota = _nonnegative_integer(min_per_branch, name="min_per_branch")
    scale_m = _positive_finite_scale(
        spatial_diversity_scale_m,
        name="spatial_diversity_scale_m",
    )
    return _LEGACY_SPATIAL_DIVERSITY_CAP_RESERVOIR(
        reservoir,
        max_candidates_per_frame=cap,
        min_per_source=source_quota,
        min_per_branch=branch_quota,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        branch_column=branch_column,
        spatial_diversity_weight=spatial_diversity_weight,
        spatial_diversity_scale_m=scale_m,
        spatial_distance_cap_m=spatial_distance_cap_m,
    )


_IMPL._score = _score
_IMPL.spatial_diversity_cap_reservoir = spatial_diversity_cap_reservoir

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_nonnegative_integer"] = _nonnegative_integer
globals()["_positive_finite_scale"] = _positive_finite_scale
globals()["_score"] = _score
globals()["spatial_diversity_cap_reservoir"] = spatial_diversity_cap_reservoir

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
