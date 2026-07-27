"""Runtime safety for radar selection and sequence-scoped cat-probability filtering."""

from __future__ import annotations

from collections.abc import Hashable
from functools import wraps
from importlib import import_module

import numpy as np
import pandas as pd

from raft_uav.io._catprob_frame_grouping import catprob_best_per_frame_rows
from raft_uav.numeric import optional_float


_aerpaw = import_module("raft_uav.io.aerpaw")
_ORIGINAL_SELECT_RADAR_MEASUREMENT_ROWS = _aerpaw.select_radar_measurement_rows


def _finite_real_control(value: object, *, name: str) -> float:
    """Return a finite real scalar control value."""

    number = optional_float(value)
    if number is None:
        raise ValueError(f"{name} must be a finite real scalar")
    return number


def _nonnegative_real_control(value: object, *, name: str) -> float:
    """Return a finite non-negative real scalar control value."""

    number = _finite_real_control(value, name=name)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _validated_selection_controls(
    radar: pd.DataFrame,
    *,
    selection: str,
    truth: pd.DataFrame | None,
    catprob_threshold: object,
    truth_gate_m: object,
    truth_time_gate_s: object,
) -> tuple[object, object, object]:
    """Validate the selection mode and controls used by a non-empty input."""

    if not isinstance(selection, str) or selection not in _aerpaw.RADAR_SELECTION_MODES:
        raise ValueError(f"unknown radar selection {selection!r}")
    if radar.empty:
        return catprob_threshold, truth_gate_m, truth_time_gate_s
    if selection in {"catprob", "catprob-all"}:
        catprob_threshold = _finite_real_control(
            catprob_threshold,
            name="catprob_threshold",
        )
    elif selection == "truth-gated" and truth is not None:
        truth_gate_m = _nonnegative_real_control(
            truth_gate_m,
            name="truth_gate_m",
        )
        truth_time_gate_s = _nonnegative_real_control(
            truth_time_gate_s,
            name="truth_time_gate_s",
        )
    return catprob_threshold, truth_gate_m, truth_time_gate_s


def _sequence_group_key(value: object) -> tuple[object, ...]:
    """Return a stable key without merging different serialized ID types."""

    if _aerpaw._is_missing_scalar(value):
        return ("missing",)
    if isinstance(value, Hashable):
        return ("value", type(value), value)
    return ("repr", type(value), repr(value))


def _sequence_position_groups(values: pd.Series) -> list[list[int]]:
    groups: dict[tuple[object, ...], list[int]] = {}
    for position, value in enumerate(values.to_numpy(dtype=object)):
        groups.setdefault(_sequence_group_key(value), []).append(position)
    return list(groups.values())


def _select_unscoped_radar_measurement_rows(
    radar: pd.DataFrame,
    *,
    selection: str,
    truth: pd.DataFrame | None,
    catprob_threshold: object,
    truth_gate_m: object,
    truth_time_gate_s: object,
) -> pd.DataFrame:
    """Dispatch one sequence through the authoritative selection implementation."""

    if selection == "catprob":
        return catprob_best_per_frame_rows(radar, float(catprob_threshold))
    return _ORIGINAL_SELECT_RADAR_MEASUREMENT_ROWS(
        radar,
        selection=selection,
        truth=truth,
        catprob_threshold=catprob_threshold,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )


@wraps(_ORIGINAL_SELECT_RADAR_MEASUREMENT_ROWS)
def _select_radar_measurement_rows(
    radar: pd.DataFrame,
    *,
    selection: str = "catprob",
    truth: pd.DataFrame | None = None,
    catprob_threshold: float = 0.5,
    truth_gate_m: float = 150.0,
    truth_time_gate_s: float = 1.0,
) -> pd.DataFrame:
    """Validate controls and keep at most one catprob candidate per sequence frame."""

    catprob_threshold, truth_gate_m, truth_time_gate_s = _validated_selection_controls(
        radar,
        selection=selection,
        truth=truth,
        catprob_threshold=catprob_threshold,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )

    if selection != "catprob" or "sequence_id" not in radar.columns or radar.empty:
        return _select_unscoped_radar_measurement_rows(
            radar,
            selection=selection,
            truth=truth,
            catprob_threshold=catprob_threshold,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )

    position_groups = _sequence_position_groups(radar["sequence_id"])
    if len(position_groups) <= 1:
        return _select_unscoped_radar_measurement_rows(
            radar,
            selection=selection,
            truth=truth,
            catprob_threshold=catprob_threshold,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )

    order_column = "__raft_uav_catprob_input_order"
    while order_column in radar.columns:
        order_column = f"_{order_column}"
    scoped = radar.copy()
    scoped[order_column] = np.arange(len(scoped), dtype=int)

    selected_parts = [
        _select_unscoped_radar_measurement_rows(
            scoped.iloc[positions],
            selection=selection,
            truth=truth,
            catprob_threshold=catprob_threshold,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )
        for positions in position_groups
    ]
    selected = pd.concat(selected_parts, axis=0, sort=False)
    if selected.empty:
        return radar.iloc[0:0].copy()
    return (
        selected.sort_values(order_column, kind="mergesort")
        .drop(columns=[order_column])
        .copy()
    )


def _install_class_probability_label_validation() -> None:
    from raft_uav.mmuad._class_probability_label_validation_patch import (
        install as install_label_validation,
    )

    install_label_validation()


def install() -> None:
    """Install class-probability runtime fixes once per interpreter."""

    if not getattr(_aerpaw, "_catprob_sequence_patch_applied", False):
        _aerpaw._catprob_best_per_frame_rows = catprob_best_per_frame_rows
        _aerpaw.select_radar_measurement_rows = _select_radar_measurement_rows
        legacy = getattr(_aerpaw, "_IMPL", None)
        if legacy is not None:
            legacy._catprob_best_per_frame_rows = catprob_best_per_frame_rows
            legacy.select_radar_measurement_rows = _select_radar_measurement_rows
        _aerpaw._catprob_sequence_patch_applied = True
    _install_class_probability_label_validation()
