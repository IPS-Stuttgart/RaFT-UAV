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
_legacy = getattr(_aerpaw, "_IMPL", None)
_ORIGINAL_TRUTH_GATED_ROWS = getattr(
    _legacy if _legacy is not None else _aerpaw,
    "_truth_gated_rows",
)


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


def _truth_with_final_duplicate_samples(truth: pd.DataFrame) -> pd.DataFrame:
    """Keep the final row for every finite numeric-equivalent truth timestamp."""

    if truth.empty or "time_s" not in truth.columns:
        return truth

    time_keys = pd.to_numeric(truth["time_s"], errors="coerce").to_numpy(dtype=float)
    final_positions: dict[float, int] = {}
    for position, time_s in enumerate(time_keys):
        if np.isfinite(time_s):
            final_positions[float(time_s)] = position

    keep_positions = [
        position
        for position, time_s in enumerate(time_keys)
        if not np.isfinite(time_s) or final_positions[float(time_s)] == position
    ]
    return truth.iloc[keep_positions].copy()


@wraps(_ORIGINAL_TRUTH_GATED_ROWS)
def _truth_gated_rows(
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    truth_gate_m: float,
    truth_time_gate_s: float,
) -> pd.DataFrame:
    """Gate radar rows against the authoritative final same-time truth sample."""

    return _ORIGINAL_TRUTH_GATED_ROWS(
        radar,
        _truth_with_final_duplicate_samples(truth),
        truth_gate_m,
        truth_time_gate_s,
    )


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


def _sequence_match_key(value: object) -> tuple[object, ...]:
    """Return a cross-table key for radar/truth sequence matching."""

    if _aerpaw._is_missing_scalar(value):
        return ("missing",)
    return ("text", str(value))


def _sequence_match_identity(value: object) -> tuple[object, ...]:
    """Return a representation-aware identity used to detect text-key collisions."""

    if _aerpaw._is_missing_scalar(value):
        return ("missing",)
    if isinstance(value, (str, np.str_)):
        return ("string", str(value))
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, (int, np.integer)):
        return ("integer", int(value))
    if isinstance(value, (float, np.floating)):
        return ("real", float(value))
    return _sequence_group_key(value)


def _sequence_positions_by_match_key(
    values: pd.Series,
    *,
    table_name: str,
) -> dict[tuple[object, ...], list[int]]:
    """Group positions by cross-table key without silently merging distinct IDs."""

    groups: dict[tuple[object, ...], list[int]] = {}
    identities: dict[tuple[object, ...], tuple[object, ...]] = {}
    representatives: dict[tuple[object, ...], object] = {}
    for position, value in enumerate(values.to_numpy(dtype=object)):
        match_key = _sequence_match_key(value)
        identity = _sequence_match_identity(value)
        if match_key in identities and identities[match_key] != identity:
            raise ValueError(
                f"{table_name} sequence_id contains ambiguous values after text "
                f"normalization: {representatives[match_key]!r} and {value!r} "
                f"both map to {str(value)!r}"
            )
        identities.setdefault(match_key, identity)
        representatives.setdefault(match_key, value)
        groups.setdefault(match_key, []).append(position)
    return groups


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
    """Validate controls and preserve optional sequence boundaries during selection."""

    catprob_threshold, truth_gate_m, truth_time_gate_s = _validated_selection_controls(
        radar,
        selection=selection,
        truth=truth,
        catprob_threshold=catprob_threshold,
        truth_gate_m=truth_gate_m,
        truth_time_gate_s=truth_time_gate_s,
    )

    sequence_scoped_catprob = (
        selection == "catprob" and "sequence_id" in radar.columns and not radar.empty
    )
    sequence_scoped_truth_gate = (
        selection == "truth-gated"
        and truth is not None
        and "sequence_id" in radar.columns
        and "sequence_id" in truth.columns
        and not radar.empty
    )
    if not sequence_scoped_catprob and not sequence_scoped_truth_gate:
        return _select_unscoped_radar_measurement_rows(
            radar,
            selection=selection,
            truth=truth,
            catprob_threshold=catprob_threshold,
            truth_gate_m=truth_gate_m,
            truth_time_gate_s=truth_time_gate_s,
        )

    if sequence_scoped_truth_gate:
        radar_groups = _sequence_positions_by_match_key(
            radar["sequence_id"],
            table_name="radar",
        )
        truth_groups = _sequence_positions_by_match_key(
            truth["sequence_id"],
            table_name="truth",
        )
        position_groups = list(radar_groups.items())
    else:
        truth_groups = {}
        position_groups = [
            (None, positions)
            for positions in _sequence_position_groups(radar["sequence_id"])
        ]
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

    selected_parts: list[pd.DataFrame] = []
    for sequence_key, positions in position_groups:
        scoped_truth = truth
        if sequence_scoped_truth_gate:
            truth_positions = truth_groups.get(sequence_key, [])
            if not truth_positions:
                continue
            scoped_truth = truth.iloc[truth_positions]
        selected_parts.append(
            _select_unscoped_radar_measurement_rows(
                scoped.iloc[positions],
                selection=selection,
                truth=scoped_truth,
                catprob_threshold=catprob_threshold,
                truth_gate_m=truth_gate_m,
                truth_time_gate_s=truth_time_gate_s,
            )
        )

    if not selected_parts:
        return radar.iloc[0:0].copy()
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
        _aerpaw._truth_gated_rows = _truth_gated_rows
        _aerpaw.select_radar_measurement_rows = _select_radar_measurement_rows
        legacy = getattr(_aerpaw, "_IMPL", None)
        if legacy is not None:
            legacy._catprob_best_per_frame_rows = catprob_best_per_frame_rows
            legacy._truth_gated_rows = _truth_gated_rows
            legacy.select_radar_measurement_rows = _select_radar_measurement_rows
        _aerpaw._catprob_sequence_patch_applied = True
    _install_class_probability_label_validation()
