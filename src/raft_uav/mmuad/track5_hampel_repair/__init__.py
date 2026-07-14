"""Compatibility wrapper that preserves Track 5 Hampel-repair endpoints.

The maintained implementation lives in the sibling ``track5_hampel_repair.py``
module. This package preserves the public import path while ensuring that only
interior trajectory rows are eligible for local-median replacement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_hampel_repair.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_hampel_repair_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load Hampel repair implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _repair_xyz_once(
    work: pd.DataFrame,
    xyz: np.ndarray,
    original_xyz: np.ndarray,
    *,
    window_radius: int,
    sigma_threshold: float,
    min_scale_m: float,
    min_residual_m: float,
    repair_blend: float,
    iteration: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Repair eligible interior rows while leaving both endpoints unchanged."""

    out = xyz.copy()
    diagnostics: list[dict[str, Any]] = []
    for index in range(len(xyz)):
        start = max(0, index - window_radius)
        stop = min(len(xyz), index + window_radius + 1)
        neighbor_indices = [item for item in range(start, stop) if item != index]
        neighbors = xyz[neighbor_indices]
        neighbors = neighbors[np.isfinite(neighbors).all(axis=1)]
        local_median = np.full(3, np.nan, dtype=float)
        robust_scale_m = np.nan
        residual_m = np.nan
        threshold_m = np.nan
        applied = False
        is_interior = 0 < index < len(xyz) - 1
        if is_interior and len(neighbors) >= 2 and np.isfinite(xyz[index]).all():
            local_median = np.median(neighbors, axis=0)
            neighbor_residuals = np.linalg.norm(neighbors - local_median, axis=1)
            robust_scale_m = max(
                float(np.median(neighbor_residuals) * 1.4826),
                min_scale_m,
            )
            residual_m = float(np.linalg.norm(xyz[index] - local_median))
            threshold_m = max(
                float(min_residual_m),
                float(sigma_threshold) * robust_scale_m,
            )
            if residual_m > threshold_m:
                out[index] = (
                    (1.0 - repair_blend) * xyz[index]
                    + repair_blend * local_median
                )
                applied = True
        diagnostics.append(
            {
                "sequence_id": work.loc[index, "sequence_id"],
                "time_s": float(work.loc[index, "time_s"]),
                "iteration": int(iteration),
                "local_window_start_index": int(start),
                "local_window_stop_index": int(stop),
                "local_neighbor_count": int(len(neighbors)),
                "local_median_x_m": (
                    float(local_median[0]) if np.isfinite(local_median[0]) else np.nan
                ),
                "local_median_y_m": (
                    float(local_median[1]) if np.isfinite(local_median[1]) else np.nan
                ),
                "local_median_z_m": (
                    float(local_median[2]) if np.isfinite(local_median[2]) else np.nan
                ),
                "hampel_residual_m": residual_m,
                "hampel_scale_m": robust_scale_m,
                "hampel_threshold_m": threshold_m,
                "hampel_iteration_applied": bool(applied),
                "original_state_x_m": float(original_xyz[index, 0]),
                "original_state_y_m": float(original_xyz[index, 1]),
                "original_state_z_m": float(original_xyz[index, 2]),
            }
        )
    return out, pd.DataFrame.from_records(
        diagnostics,
        columns=_IMPL._diagnostic_columns(),
    )


_IMPL._repair_xyz_once = _repair_xyz_once

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_repair_xyz_once"] = _repair_xyz_once

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
