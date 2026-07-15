"""Compatibility fixes for exact pooled radar time-offset statistics.

The maintained implementation lives in the sibling ``time_offset.py`` module.
This package preserves the public import path while pooling per-frame oracle
errors before computing nonlinear aggregate statistics such as standard
deviation and percentiles.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.evaluation.radar_oracle_diagnostics import nearest_candidate_oracle

_IMPL_PATH = Path(__file__).resolve().parent.parent / "time_offset.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._time_offset_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load time-offset calibration implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def aggregate_radar_time_offset_sweep(
    training_pairs: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    offsets_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> pd.DataFrame:
    """Aggregate radar oracle errors exactly across all training flights."""

    rows: list[dict[str, float]] = []
    for offset in (float(value) for value in offsets_s):
        errors_3d: list[np.ndarray] = []
        errors_2d: list[np.ndarray] = []
        frame_count = 0
        for radar, truth in training_pairs:
            if radar.empty or truth.empty:
                continue
            frame_count += _IMPL._radar_frame_count(radar)
            selected = nearest_candidate_oracle(
                radar,
                truth,
                time_offset_s=offset,
                max_time_delta_s=max_time_delta_s,
            )
            if selected.empty:
                continue
            selected_3d = pd.to_numeric(
                selected["oracle_error_3d_m"],
                errors="coerce",
            ).to_numpy(dtype=float)
            selected_2d = pd.to_numeric(
                selected["oracle_error_2d_m"],
                errors="coerce",
            ).to_numpy(dtype=float)
            errors_3d.append(selected_3d[np.isfinite(selected_3d)])
            errors_2d.append(selected_2d[np.isfinite(selected_2d)])

        pooled_3d = _IMPL._concat(errors_3d)
        pooled_2d = _IMPL._concat(errors_2d)
        row = {
            "time_offset_s": offset,
            "count": float(pooled_3d.size),
            "coverage": _IMPL._coverage(pooled_3d.size, frame_count),
        }
        row.update(_IMPL._stats(pooled_3d, "3d"))
        row.update(_IMPL._stats(pooled_2d, "2d"))
        rows.append(row)
    return pd.DataFrame.from_records(
        rows,
        columns=["time_offset_s", *_IMPL.PAPER_METRIC_COLUMNS],
    )


_IMPL.aggregate_radar_time_offset_sweep = aggregate_radar_time_offset_sweep

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["aggregate_radar_time_offset_sweep"] = aggregate_radar_time_offset_sweep

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
