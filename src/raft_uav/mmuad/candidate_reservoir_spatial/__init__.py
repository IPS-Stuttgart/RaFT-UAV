"""Compatibility package validating spatial-diversity reservoir scales.

The maintained implementation lives in the sibling
``candidate_reservoir_spatial.py`` module. This package preserves the public
import path while rejecting malformed spatial scales before candidate selection.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

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


def spatial_diversity_cap_reservoir(
    reservoir: pd.DataFrame,
    *,
    max_candidates_per_frame: int = 40,
    min_per_source: int = 1,
    min_per_branch: int = 1,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "confidence",
    branch_column: str = "candidate_branch",
    spatial_diversity_weight: float = 1.0,
    spatial_diversity_scale_m: float = 10.0,
    spatial_distance_cap_m: float = 50.0,
) -> pd.DataFrame:
    """Cap candidates after validating the spatial decay scale."""

    scale_m = _positive_finite_scale(
        spatial_diversity_scale_m,
        name="spatial_diversity_scale_m",
    )
    return _LEGACY_SPATIAL_DIVERSITY_CAP_RESERVOIR(
        reservoir,
        max_candidates_per_frame=max_candidates_per_frame,
        min_per_source=min_per_source,
        min_per_branch=min_per_branch,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        branch_column=branch_column,
        spatial_diversity_weight=spatial_diversity_weight,
        spatial_diversity_scale_m=scale_m,
        spatial_distance_cap_m=spatial_distance_cap_m,
    )


_IMPL._positive_finite_scale = _positive_finite_scale
_IMPL.spatial_diversity_cap_reservoir = spatial_diversity_cap_reservoir

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_positive_finite_scale"] = _positive_finite_scale
globals()["spatial_diversity_cap_reservoir"] = spatial_diversity_cap_reservoir

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
