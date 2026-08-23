"""Keep RF burst stress perturbations scoped to physical flights."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import perturbations as _IMPL

_SCOPE_COLUMNS = ("sequence_id", "flight_id")


def _scope_columns(frame: pd.DataFrame) -> list[str]:
    """Return all available physical-scope columns in stable order."""

    return [column for column in _SCOPE_COLUMNS if column in frame.columns]


def drop_rf_bursts(
    frame: pd.DataFrame,
    *,
    rate: float,
    rng: Any,
) -> pd.DataFrame:
    """Drop five-second RF bursts independently within each physical flight."""

    drop_rate = _IMPL._drop_rate(rate, name="rate")
    if frame.empty or drop_rate == 0.0 or "time_s" not in frame.columns:
        return frame.copy()

    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy(dtype=float)
    if times.size == 0:
        return frame.copy()
    valid_time_mask = np.isfinite(times)
    if not np.any(valid_time_mask):
        return frame.copy()

    finite_rows = frame.loc[valid_time_mask].copy()
    finite_rows["_stress_time_s"] = times[valid_time_mask]
    scope_columns = _scope_columns(finite_rows)
    group_columns = [*scope_columns, "_stress_burst_bin"]
    if scope_columns:
        scope_start = finite_rows.groupby(
            scope_columns,
            sort=False,
            dropna=False,
        )["_stress_time_s"].transform("min")
    else:
        scope_start = float(finite_rows["_stress_time_s"].min())
    finite_rows["_stress_burst_bin"] = np.floor(
        (finite_rows["_stress_time_s"] - scope_start) / 5.0
    ).astype(int)
    group_ids = (
        finite_rows.groupby(group_columns, sort=True, dropna=False)
        .ngroup()
        .to_numpy()
    )
    groups = np.unique(group_ids)
    dropped = set(groups[rng.random(len(groups)) < drop_rate].tolist())

    keep_mask = np.ones(times.shape, dtype=bool)
    keep_mask[valid_time_mask] = ~np.isin(group_ids, list(dropped))
    return frame.loc[keep_mask].copy()


_IMPL.drop_rf_bursts = drop_rf_bursts
