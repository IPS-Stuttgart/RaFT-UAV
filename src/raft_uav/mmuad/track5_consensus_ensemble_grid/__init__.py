"""Compatibility fix for one-pass Track 5 consensus-grid iterables.

The maintained implementation lives in the sibling
``track5_consensus_ensemble_grid.py`` module. This package preserves the public
import path while materializing each documented parameter-grid iterable once so
nested search loops evaluate the complete Cartesian product.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_consensus_ensemble_grid.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_consensus_ensemble_grid_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load consensus-grid implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SEARCH = _IMPL.search_track5_consensus_ensemble_grid


def search_track5_consensus_ensemble_grid(
    estimate_inputs: Iterable[EstimateInput],
    *,
    template: pd.DataFrame,
    truth: pd.DataFrame,
    consensus_radius_m: Iterable[float] = (2.0, 5.0, 10.0),
    min_consensus_weight_fraction: Iterable[float] = (0.0, 0.5),
    fallback_policy: Iterable[str] = ("max-weight", "weighted-mean"),
    max_nearest_time_delta_s: float | None = None,
    selection_objective: str = "pooled-mse",
    sequence_objective_weight: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate every grid combination when callers supply one-pass iterables."""

    radius_values = tuple(consensus_radius_m)
    min_fraction_values = tuple(min_consensus_weight_fraction)
    fallback_values = tuple(fallback_policy)
    return _ORIGINAL_SEARCH(
        estimate_inputs,
        template=template,
        truth=truth,
        consensus_radius_m=radius_values,
        min_consensus_weight_fraction=min_fraction_values,
        fallback_policy=fallback_values,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        selection_objective=selection_objective,
        sequence_objective_weight=sequence_objective_weight,
    )


_IMPL.search_track5_consensus_ensemble_grid = search_track5_consensus_ensemble_grid

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["search_track5_consensus_ensemble_grid"] = (
    search_track5_consensus_ensemble_grid
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
