"""Runtime fix for sequence-scoped radar cat-probability selection."""

from __future__ import annotations

from collections.abc import Hashable
from functools import wraps
from importlib import import_module

import numpy as np
import pandas as pd

_aerpaw = import_module("raft_uav.io.aerpaw")
_ORIGINAL_SELECT_RADAR_MEASUREMENT_ROWS = _aerpaw.select_radar_measurement_rows


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
    """Keep at most one cat-probability candidate per frame and sequence."""

    if selection != "catprob" or "sequence_id" not in radar.columns or radar.empty:
        return _ORIGINAL_SELECT_RADAR_MEASUREMENT_ROWS(
            radar,
            selection=selection,
            truth=truth,
            catprob_threshold=catprob_threshold,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )

    position_groups = _sequence_position_groups(radar["sequence_id"])
    if len(position_groups) <= 1:
        return _ORIGINAL_SELECT_RADAR_MEASUREMENT_ROWS(
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
        _ORIGINAL_SELECT_RADAR_MEASUREMENT_ROWS(
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
        _aerpaw.select_radar_measurement_rows = _select_radar_measurement_rows
        legacy = getattr(_aerpaw, "_IMPL", None)
        if legacy is not None:
            legacy.select_radar_measurement_rows = _select_radar_measurement_rows
        _aerpaw._catprob_sequence_patch_applied = True
    _install_class_probability_label_validation()
