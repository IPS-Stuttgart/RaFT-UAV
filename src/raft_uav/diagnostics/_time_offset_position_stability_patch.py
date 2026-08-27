"""Keep time-offset position errors stable for large finite coordinates."""

from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module

import numpy as np
import pandas as pd


_time_offset = import_module("raft_uav.diagnostics.time_offset")
_legacy = getattr(_time_offset, "_legacy", _time_offset)
_PATCH_MARKER = "_raft_uav_time_offset_position_stability_patch_applied"


def _stable_position_errors(
    positions: np.ndarray,
    truth_positions: np.ndarray,
    *,
    dimensions: int,
) -> np.ndarray:
    """Return row-wise Euclidean errors without squaring large components."""

    position_matrix = np.asarray(positions, dtype=float)
    truth_matrix = np.asarray(truth_positions, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        deltas = (
            position_matrix[:, :dimensions]
            - truth_matrix[:, :dimensions]
        )
        return np.hypot.reduce(np.abs(deltas), axis=1)


def sweep_positions_against_truth(
    *,
    measurement_times_s: np.ndarray,
    measurement_positions_m: np.ndarray,
    truth: pd.DataFrame,
    taus_s: Iterable[float],
    dimensions: int,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Sweep offsets without overflowing representable Euclidean errors."""

    times = np.asarray(measurement_times_s, dtype=float).reshape(-1)
    positions = np.asarray(measurement_positions_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] < dimensions:
        raise ValueError("measurement_positions_m must be shape (n, >=dimensions)")
    if positions.shape[0] != times.size:
        raise ValueError("measurement times and positions must have the same length")

    rows = []
    for tau_s in taus_s:
        shifted_times = times + float(tau_s)
        truth_positions, mask = _time_offset.truth_positions_at_times(
            truth,
            shifted_times,
            max_delta_s=max_truth_time_delta_s,
        )
        finite = (
            mask
            & np.isfinite(positions[:, :dimensions]).all(axis=1)
        )
        errors = _stable_position_errors(
            positions[finite],
            truth_positions[finite],
            dimensions=dimensions,
        )
        rows.append(
            _time_offset.summarize_errors(
                tau_s=float(tau_s),
                candidate_count=int(times.size),
                selected_count=int(times.size),
                matched_count=int(errors.size),
                errors_m=errors,
            )
        )
    return pd.DataFrame.from_records(rows)


def sweep_radar_against_truth(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    taus_s: Iterable[float],
    dimensions: int,
    selection: str,
    catprob_threshold: float,
    max_truth_time_delta_s: float,
) -> pd.DataFrame:
    """Sweep radar offsets without overflowing representable position errors."""

    groups = _time_offset.radar_frame_groups(radar)
    longest_track_id = (
        _time_offset._longest_track_id(radar)
        if selection == "longest-track"
        else None
    )
    rows = []
    for tau_s in taus_s:
        selected_times, selected_positions = (
            _time_offset.select_radar_rows_for_offset(
                groups=groups,
                truth=truth,
                tau_s=float(tau_s),
                selection=selection,
                catprob_threshold=catprob_threshold,
                longest_track_id=longest_track_id,
                max_truth_time_delta_s=max_truth_time_delta_s,
            )
        )
        if len(selected_times):
            truth_positions, mask = _time_offset.truth_positions_at_times(
                truth,
                np.asarray(selected_times, dtype=float),
                max_delta_s=max_truth_time_delta_s,
            )
            positions = np.asarray(selected_positions, dtype=float)
            finite = (
                mask
                & np.isfinite(positions[:, :dimensions]).all(axis=1)
            )
            errors = _stable_position_errors(
                positions[finite],
                truth_positions[finite],
                dimensions=dimensions,
            )
        else:
            errors = np.empty(0, dtype=float)
        rows.append(
            _time_offset.summarize_errors(
                tau_s=float(tau_s),
                candidate_count=len(groups),
                selected_count=len(selected_times),
                matched_count=int(errors.size),
                errors_m=errors,
            )
        )
    return pd.DataFrame.from_records(rows)


def nearest_candidate_to_truth(
    candidates: pd.DataFrame,
    truth_position: np.ndarray | None,
) -> pd.Series | None:
    """Select the nearest finite candidate with an overflow-stable norm."""

    if truth_position is None or candidates.empty:
        return None
    sanitizer = getattr(_time_offset, "_finite_position_candidates", None)
    if sanitizer is not None:
        candidates = sanitizer(candidates)
    if candidates.empty:
        return None

    candidate_xyz = candidates[["east_m", "north_m", "up_m"]].to_numpy(
        dtype=float
    )
    truth_xyz = np.asarray(truth_position, dtype=float).reshape(1, 3)
    errors = _stable_position_errors(
        candidate_xyz,
        np.broadcast_to(truth_xyz, candidate_xyz.shape),
        dimensions=3,
    )
    if not np.isfinite(errors).any():
        return None
    return candidates.iloc[int(np.nanargmin(errors))].copy()


def install() -> None:
    """Install stable time-offset position computations once per interpreter."""

    if getattr(_time_offset, _PATCH_MARKER, False):
        return
    for module in (_legacy, _time_offset):
        module.sweep_positions_against_truth = sweep_positions_against_truth
        module.sweep_radar_against_truth = sweep_radar_against_truth
        module.nearest_candidate_to_truth = nearest_candidate_to_truth
        setattr(module, _PATCH_MARKER, True)


install()
