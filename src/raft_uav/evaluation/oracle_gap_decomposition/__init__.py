"""Compatibility fixes for oracle-gap and confidence diagnostics.

The maintained implementation lives in the sibling
``oracle_gap_decomposition.py`` module. This package preserves the public import
path while keeping estimate columns, row order, and invalid-time rows intact
when selected-radar context is attached, scoping that context by sequence when
sequence metadata is available, preserving partially indexed radar frames,
preventing non-finite frame times from matching arbitrary truth or estimate
rows, requiring exact integer radar track identifiers in diagnostic outputs,
and rejecting invalid oracle-gap thresholds.
"""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "oracle_gap_decomposition.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._oracle_gap_decomposition_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load oracle-gap decomposition implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP = _IMPL.decompose_radar_oracle_gap
_ORIGINAL_NEAREST_POSITION = _IMPL._nearest_position
_ORIGINAL_NEAREST_ESTIMATE_ERROR = _IMPL._nearest_estimate_error
_ORACLE_GAP_THRESHOLD_NAMES = (
    "plausible_candidate_gate_m",
    "truth_time_gate_s",
    "estimate_time_gate_s",
    "drift_error_gate_m",
)
_CONTEXT_COLUMNS = (
    "track_id",
    "association_score",
    "association_nis",
    "association_weight_entropy",
    "association_hypothesis_count",
)
_MISSING_SEQUENCE_ID_STRINGS = frozenset({"", "nan", "none", "<na>", "nat"})
_ROW_ORDER_COLUMN = "__raft_uav_confidence_row_order"
_MERGE_TIME_COLUMN = "__raft_uav_confidence_merge_time_s"
_SEQUENCE_KEY_COLUMN = "__raft_uav_confidence_sequence_id"


def _oracle_gap_config_post_init(config: object) -> None:
    """Require every oracle-gap threshold to be a finite positive scalar."""

    for name in _ORACLE_GAP_THRESHOLD_NAMES:
        value = optional_float(getattr(config, name))
        if value is None:
            raise ValueError(f"{name} must be a finite number")
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")


def _radar_frame_groups(radar: pd.DataFrame) -> list[pd.DataFrame]:
    """Group indexed frames exactly and fall back per row to finite timestamps."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id", "track_index")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns).reset_index(drop=True)
    times = pd.to_numeric(ordered["time_s"], errors="coerce")
    if "frame_index" in ordered.columns:
        frame_indices = pd.to_numeric(ordered["frame_index"], errors="coerce")
    else:
        frame_indices = pd.Series(np.nan, index=ordered.index, dtype=float)
    group_keys = pd.Series(
        [
            ("frame_index", float(frame_index))
            if np.isfinite(frame_index)
            else ("time_s", float(time_s))
            if np.isfinite(time_s)
            else None
            for frame_index, time_s in zip(frame_indices, times, strict=True)
        ],
        index=ordered.index,
        dtype=object,
    )
    usable = group_keys.notna()
    ordered = ordered.loc[usable]
    group_keys = group_keys.loc[usable]
    return [group.copy() for _, group in ordered.groupby(group_keys, sort=False)]


@wraps(_ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP)
def decompose_radar_oracle_gap(*args, **kwargs) -> pd.DataFrame:
    """Run the maintained decomposition without warning on all-missing frame times."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Mean of empty slice",
            category=RuntimeWarning,
        )
        return _ORIGINAL_DECOMPOSE_RADAR_ORACLE_GAP(*args, **kwargs)


def _nearest_position(
    frame: pd.DataFrame,
    *,
    time_s: float,
    max_delta_s: float,
) -> np.ndarray | None:
    """Reject non-finite query times instead of selecting an arbitrary row."""

    normalized_time_s = optional_float(time_s)
    if normalized_time_s is None:
        return None
    return _ORIGINAL_NEAREST_POSITION(
        frame,
        time_s=normalized_time_s,
        max_delta_s=max_delta_s,
    )


def _nearest_estimate_error(
    estimate_times: np.ndarray,
    estimate_positions: np.ndarray,
    *,
    time_s: float,
    truth_position: np.ndarray,
    max_delta_s: float,
) -> float:
    """Return no estimate error for a non-finite query timestamp."""

    normalized_time_s = optional_float(time_s)
    if normalized_time_s is None:
        return float("nan")
    return _ORIGINAL_NEAREST_ESTIMATE_ERROR(
        estimate_times,
        estimate_positions,
        time_s=normalized_time_s,
        truth_position=truth_position,
        max_delta_s=max_delta_s,
    )


def _normalized_context_sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed sequence keys while preserving missing identifiers."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    missing = keys.isna() | keys.str.casefold().isin(_MISSING_SEQUENCE_ID_STRINGS)
    return keys.mask(missing)


def _merge_selected_context(
    estimates: pd.DataFrame,
    selected: pd.DataFrame,
) -> pd.DataFrame:
    """Attach selected-radar context without crossing sequence boundaries."""

    if "time_s" not in estimates.columns or "time_s" not in selected.columns:
        return estimates

    available_context = [
        column for column in _CONTEXT_COLUMNS if column in selected.columns
    ]
    if not available_context:
        return estimates

    scope_by_sequence = (
        "sequence_id" in estimates.columns and "sequence_id" in selected.columns
    )
    left = estimates.copy()
    original_index = left.index
    left[_ROW_ORDER_COLUMN] = np.arange(len(left), dtype=np.int64)
    left[_MERGE_TIME_COLUMN] = pd.to_numeric(left["time_s"], errors="coerce")
    if scope_by_sequence:
        left[_SEQUENCE_KEY_COLUMN] = _normalized_context_sequence_keys(
            left["sequence_id"]
        )

    context = selected[["time_s", *available_context]].copy()
    context[_MERGE_TIME_COLUMN] = pd.to_numeric(
        context["time_s"],
        errors="coerce",
    )
    if scope_by_sequence:
        context[_SEQUENCE_KEY_COLUMN] = _normalized_context_sequence_keys(
            selected["sequence_id"]
        )
    renamed_context = {
        column: f"selected_context_{column}" for column in available_context
    }
    context = context.drop(columns=["time_s"]).rename(columns=renamed_context)
    usable_context = np.isfinite(
        context[_MERGE_TIME_COLUMN].to_numpy(dtype=float)
    )
    if scope_by_sequence:
        usable_context &= context[_SEQUENCE_KEY_COLUMN].notna().to_numpy(dtype=bool)
    context = context.loc[usable_context].sort_values(
        _MERGE_TIME_COLUMN,
        kind="mergesort",
    )

    output_columns = list(renamed_context.values())
    valid_left = np.isfinite(left[_MERGE_TIME_COLUMN].to_numpy(dtype=float))
    if scope_by_sequence:
        valid_left &= left[_SEQUENCE_KEY_COLUMN].notna().to_numpy(dtype=bool)
    if context.empty or not valid_left.any():
        for column in output_columns:
            left[column] = np.nan
        merged = left
    else:
        sortable_left = left.loc[valid_left].sort_values(
            _MERGE_TIME_COLUMN,
            kind="mergesort",
        )
        if scope_by_sequence:
            matched = pd.merge_asof(
                sortable_left,
                context,
                on=_MERGE_TIME_COLUMN,
                by=_SEQUENCE_KEY_COLUMN,
                direction="nearest",
                tolerance=0.25,
            )
        else:
            matched = pd.merge_asof(
                sortable_left,
                context,
                on=_MERGE_TIME_COLUMN,
                direction="nearest",
                tolerance=0.25,
            )
        unmatched = left.loc[~valid_left].copy()
        for column in output_columns:
            unmatched[column] = np.nan
        merged = pd.concat([matched, unmatched], ignore_index=True, sort=False)

    helper_columns = [_ROW_ORDER_COLUMN, _MERGE_TIME_COLUMN]
    if scope_by_sequence:
        helper_columns.append(_SEQUENCE_KEY_COLUMN)
    merged = merged.sort_values(_ROW_ORDER_COLUMN).drop(columns=helper_columns)
    merged.index = original_index
    return merged


def selected_track_stability_metrics(
    selected_radar: pd.DataFrame | None,
) -> dict[str, object]:
    """Return identity stability using only exact integer radar track IDs."""

    if (
        selected_radar is None
        or selected_radar.empty
        or "track_id" not in selected_radar.columns
    ):
        return {
            "selected_radar_rows": 0,
            "track_switch_count": 0,
            "dominant_track_fraction": float("nan"),
            "selected_track_entropy": float("nan"),
        }
    sort_columns = [
        column
        for column in ("time_s", "frame_index")
        if column in selected_radar.columns
    ]
    ordered = selected_radar.sort_values(sort_columns) if sort_columns else selected_radar
    track_ids = pd.Series(
        [optional_int(value) for value in ordered["track_id"]],
        index=ordered.index,
        dtype="Int64",
    ).dropna()
    if track_ids.empty:
        return {
            "selected_radar_rows": int(len(ordered)),
            "track_switch_count": 0,
            "dominant_track_fraction": float("nan"),
            "selected_track_entropy": float("nan"),
        }
    values = track_ids.to_numpy(dtype=int)
    switches = int(np.count_nonzero(values[1:] != values[:-1])) if values.size > 1 else 0
    counts = track_ids.value_counts()
    probabilities = counts.to_numpy(dtype=float) / float(counts.sum())
    entropy = float(
        -np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)))
    )
    gaps = _IMPL._time_gaps_s(ordered)
    return {
        "selected_radar_rows": int(len(ordered)),
        "finite_track_id_rows": int(values.size),
        "unique_selected_track_ids": int(counts.size),
        "track_switch_count": switches,
        "track_switch_rate": _IMPL._safe_rate(switches, max(values.size - 1, 0)),
        "dominant_track_id": int(counts.index[0]),
        "dominant_track_fraction": float(counts.iloc[0] / counts.sum()),
        "selected_track_entropy": entropy,
        "selected_time_gap_p95_s": _IMPL._percentile_or_nan(gaps, 95),
        "selected_time_gap_max_s": (
            float(np.max(gaps)) if gaps.size else float("nan")
        ),
    }


def _optional_track_id(value: object) -> object:
    """Return an exact integer track ID or the established empty marker."""

    track_id = optional_int(value)
    return "" if track_id is None else track_id


_IMPL.OracleGapConfig.__post_init__ = _oracle_gap_config_post_init
_IMPL.decompose_radar_oracle_gap = decompose_radar_oracle_gap
_IMPL._radar_frame_groups = _radar_frame_groups
_IMPL._nearest_position = _nearest_position
_IMPL._nearest_estimate_error = _nearest_estimate_error
_IMPL._merge_selected_context = _merge_selected_context
_IMPL.selected_track_stability_metrics = selected_track_stability_metrics
_IMPL._optional_track_id = _optional_track_id

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["decompose_radar_oracle_gap"] = decompose_radar_oracle_gap
globals()["_radar_frame_groups"] = _radar_frame_groups
globals()["_nearest_position"] = _nearest_position
globals()["_nearest_estimate_error"] = _nearest_estimate_error
globals()["_normalized_context_sequence_keys"] = _normalized_context_sequence_keys
globals()["_merge_selected_context"] = _merge_selected_context
globals()["selected_track_stability_metrics"] = selected_track_stability_metrics
globals()["_optional_track_id"] = _optional_track_id

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
