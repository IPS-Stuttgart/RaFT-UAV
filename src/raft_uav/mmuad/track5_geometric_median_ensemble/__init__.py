"""Compatibility wrapper with a singularity-safe geometric-median solver.

The maintained implementation lives in the sibling
``track5_geometric_median_ensemble.py`` module. This package preserves the public
import path while applying the modified Weiszfeld update when an iterate
coincides with one or more candidate positions.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_geometric_median_ensemble.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_geometric_median_ensemble_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load geometric-median implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def weighted_geometric_median(
    xyz: np.ndarray,
    weights: np.ndarray,
    *,
    max_iterations: int = 64,
    tolerance_m: float = 1.0e-4,
) -> tuple[np.ndarray, int, float]:
    """Compute a weighted 3D geometric median with modified Weiszfeld updates.

    The ordinary Weiszfeld formula is singular when the current iterate equals
    an input point. Replacing that zero distance with a small constant can pin
    the solution to a non-optimal point. The modified update accounts for the
    total weight at the coincident location and moves away unless the geometric
    median optimality condition is already satisfied.
    """

    points = np.asarray(xyz, dtype=float)
    safe_weights = np.asarray(weights, dtype=float)
    finite = (
        np.isfinite(points).all(axis=1)
        & np.isfinite(safe_weights)
        & (safe_weights > 0.0)
    )
    points = points[finite]
    safe_weights = safe_weights[finite]
    if len(points) == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=float), 0, np.nan
    if len(points) == 1:
        return points[0].astype(float), 0, 0.0

    center = np.sum(safe_weights[:, None] * points, axis=0) / float(
        np.sum(safe_weights)
    )
    last_displacement = np.inf
    scale = max(
        1.0,
        float(np.linalg.norm(center)),
        float(np.max(np.linalg.norm(points, axis=1))),
    )
    coincidence_tolerance = 8.0 * np.finfo(float).eps * scale

    for iteration in range(1, int(max_iterations) + 1):
        offsets = points - center[None, :]
        distances = np.linalg.norm(offsets, axis=1)
        coincident = distances <= coincidence_tolerance
        noncoincident = ~coincident

        if not np.any(noncoincident):
            return center.astype(float), iteration - 1, 0.0

        inverse_distances = safe_weights[noncoincident] / distances[noncoincident]
        candidate = np.sum(
            inverse_distances[:, None] * points[noncoincident],
            axis=0,
        ) / float(np.sum(inverse_distances))

        if np.any(coincident):
            residual = np.sum(
                safe_weights[noncoincident, None]
                * offsets[noncoincident]
                / distances[noncoincident, None],
                axis=0,
            )
            residual_norm = float(np.linalg.norm(residual))
            coincident_weight = float(np.sum(safe_weights[coincident]))
            if residual_norm <= coincident_weight:
                return center.astype(float), iteration - 1, 0.0
            retention = min(1.0, coincident_weight / residual_norm)
            updated = (1.0 - retention) * candidate + retention * center
        else:
            updated = candidate

        last_displacement = float(np.linalg.norm(updated - center))
        center = updated
        if last_displacement <= float(tolerance_m):
            return center.astype(float), iteration, last_displacement

    return center.astype(float), int(max_iterations), last_displacement


_IMPL.weighted_geometric_median = weighted_geometric_median

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["weighted_geometric_median"] = weighted_geometric_median

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
