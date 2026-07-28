"""Use symmetric truth-sample tolerance in radar oracle interpolation."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module

import numpy as np
import pandas as pd


_radar_oracle = import_module("raft_uav.evaluation.radar_oracle_diagnostics")
_ORIGINAL_INTERPOLATE_TRUTH_POSITIONS = _radar_oracle.interpolate_truth_positions
_TRUTH_TIME_MATCH_ATOL_S = float(
    getattr(_radar_oracle, "_TRUTH_TIME_MATCH_ATOL_S", 1.0e-9)
)


def interpolate_truth_positions(
    truth: pd.DataFrame,
    query_times_s: Iterable[float],
    *,
    max_time_delta_s: float | None = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept tolerance-equivalent queries on either side of a truth sample."""

    query_times = np.asarray(list(query_times_s), dtype=float).reshape(-1)
    positions, valid = _ORIGINAL_INTERPOLATE_TRUTH_POSITIONS(
        truth,
        query_times,
        max_time_delta_s=max_time_delta_s,
    )
    unresolved = np.isfinite(query_times) & ~valid
    if not unresolved.any() or truth.empty:
        return positions, valid

    required = {"time_s", "east_m", "north_m", "up_m"}
    if not required.issubset(truth.columns):
        return positions, valid

    truth_times = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(
        dtype=float
    )
    truth_xyz = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    finite = np.isfinite(truth_times) & np.isfinite(truth_xyz).all(axis=1)
    truth_times = truth_times[finite]
    truth_xyz = truth_xyz[finite]
    if truth_times.size == 0:
        return positions, valid

    order = np.argsort(truth_times, kind="mergesort")
    truth_times = truth_times[order]
    truth_xyz = truth_xyz[order]
    for query_index in np.flatnonzero(unresolved):
        query_time = float(query_times[query_index])
        insertion = int(np.searchsorted(truth_times, query_time))
        for truth_index in (insertion, insertion - 1):
            if truth_index < 0 or truth_index >= truth_times.size:
                continue
            if not np.isclose(
                truth_times[truth_index],
                query_time,
                rtol=0.0,
                atol=_TRUTH_TIME_MATCH_ATOL_S,
            ):
                continue
            if max_time_delta_s is not None and 0.0 > float(max_time_delta_s):
                break
            positions[query_index] = truth_xyz[truth_index]
            valid[query_index] = True
            break
    return positions, valid


def install() -> None:
    """Install the interpolation fix on public and legacy module entry points."""

    if getattr(_radar_oracle, "_endpoint_tolerance_patch_applied", False):
        return
    _radar_oracle.interpolate_truth_positions = interpolate_truth_positions
    implementation = getattr(_radar_oracle, "_IMPL", None)
    if implementation is not None:
        implementation.interpolate_truth_positions = interpolate_truth_positions
    _radar_oracle._endpoint_tolerance_patch_applied = True


install()
