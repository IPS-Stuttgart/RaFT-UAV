"""Compatibility fixes for research diagnostic frame and track identifiers.

The maintained implementation lives in the sibling ``diagnostics.py`` module.
This package preserves the public import path while retaining partially indexed
radar frames, preventing fractional identifiers from being truncated, and
keeping sparse research summaries well-defined.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _optional_float
from raft_uav.numeric import optional_int as _optional_int

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.research._diagnostics_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load research diagnostics from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

_ORIGINAL_LATENCY_CURVE = _LEGACY.latency_curve
_ORIGINAL_NEAREST_TRUTH_POSITION = _LEGACY._nearest_truth_position
_ORIGINAL_TRACK_SWITCH_METRICS = _LEGACY.track_switch_metrics
_DOMAIN_SHIFT_COLUMNS = [
    "feature",
    "train_count",
    "heldout_count",
    "train_mean",
    "heldout_mean",
    "mean_shift_z",
    "train_p50",
    "heldout_p50",
    "train_p90",
    "heldout_p90",
    "ks_distance",
]
_LATENCY_CURVE_COLUMNS = [
    "latency_s",
    "truth_rows",
    "covered_truth_rows",
    "truth_coverage_rate",
    "error_3d_count",
    "error_3d_rmse_m",
    "error_3d_p95_m",
]


def _nonnegative_finite_scalar(value: object, *, name: str) -> float:
    """Return a finite non-negative scalar or raise a field-specific error."""

    normalized = _optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return normalized


def _event_index_value(value: object) -> int | float | None:
    """Normalize a finite frame index without truncating fractional values."""

    integer = _optional_int(value)
    if integer is not None:
        return integer
    return _optional_float(value)


def _radar_frame_groups(
    radar: pd.DataFrame,
) -> list[tuple[tuple[str, int | float], pd.DataFrame]]:
    """Group rows by exact frame index, falling back to time per row."""

    if radar.empty:
        return []
    sort_columns = [
        column
        for column in ("time_s", "frame_index", "track_id")
        if column in radar.columns
    ]
    ordered = radar.sort_values(sort_columns, kind="mergesort").copy()
    frame_values = (
        ordered["frame_index"].tolist()
        if "frame_index" in ordered.columns
        else [None] * len(ordered)
    )
    time_values = (
        ordered["time_s"].tolist()
        if "time_s" in ordered.columns
        else [None] * len(ordered)
    )

    group_keys: list[tuple[str, int | float] | None] = []
    for frame_index, time_s in zip(frame_values, time_values, strict=True):
        event_index = _event_index_value(frame_index)
        if event_index is not None:
            group_keys.append(("frame_index", event_index))
            continue
        event_time = _optional_float(time_s)
        group_keys.append(
            None if event_time is None else ("time_s", round(event_time, 9))
        )

    key_column = "_research_diagnostic_frame_key"
    ordered[key_column] = group_keys
    ordered = ordered.loc[ordered[key_column].notna()].copy()

    groups: list[tuple[tuple[str, int | float], pd.DataFrame]] = []
    for event_key, group in ordered.groupby(key_column, sort=False):
        groups.append(
            (
                event_key,
                group.drop(columns=key_column).copy(),
            )
        )
    return groups


def _radar_event_key(frame: pd.DataFrame) -> tuple[str, int | float]:
    """Return the exact frame key used by ``_radar_frame_groups``."""

    if "frame_index" in frame.columns and not frame.empty:
        for value in frame["frame_index"]:
            event_index = _event_index_value(value)
            if event_index is not None:
                return ("frame_index", event_index)
    return ("time_s", round(float(frame["time_s"].median()), 9))


def _row_event_key(row: pd.Series) -> tuple[str, int | float]:
    """Return an exact selected-row key without integer truncation."""

    if "frame_index" in row.index:
        event_index = _event_index_value(row.get("frame_index"))
        if event_index is not None:
            return ("frame_index", event_index)
    return ("time_s", round(float(row["time_s"]), 9))


def _nearest_truth_position(
    truth: pd.DataFrame,
    *,
    time_s: float,
    max_time_delta_s: float,
):
    """Reject non-finite query times instead of selecting an arbitrary truth row."""

    normalized_time_s = _optional_float(time_s)
    if normalized_time_s is None:
        return None, float("nan")
    return _ORIGINAL_NEAREST_TRUTH_POSITION(
        truth,
        time_s=normalized_time_s,
        max_time_delta_s=max_time_delta_s,
    )


def _dense_track_id_surrogates(values: pd.Series) -> pd.Series:
    """Map exact track IDs to small nullable integers without float coercion."""

    surrogates: dict[int, int] = {}
    dense_values: list[int | None] = []
    for value in values:
        track_id = _optional_int(value)
        if track_id is None:
            dense_values.append(None)
            continue
        if track_id not in surrogates:
            surrogates[track_id] = len(surrogates)
        dense_values.append(surrogates[track_id])
    return pd.Series(dense_values, index=values.index, dtype="Int64")


def track_switch_metrics(
    selected: pd.DataFrame,
    *,
    long_gap_s: float = 5.0,
) -> dict[str, object]:
    """Compute switch metrics without inventing or merging track identities."""

    normalized = selected.copy()
    if "track_id" in normalized.columns:
        normalized["track_id"] = _dense_track_id_surrogates(normalized["track_id"])
    return _ORIGINAL_TRACK_SWITCH_METRICS(normalized, long_gap_s=long_gap_s)


def latency_curve(
    estimates_by_latency: Mapping[float | str, pd.DataFrame],
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float = 2.0,
) -> pd.DataFrame:
    """Evaluate latency tradeoffs with a stable empty-result contract."""

    normalized_gate = _nonnegative_finite_scalar(
        max_time_delta_s,
        name="max_time_delta_s",
    )
    if not estimates_by_latency:
        _LEGACY._require_columns(
            truth,
            {"time_s", *_LEGACY.PositionColumns},
            "truth",
        )
        return pd.DataFrame(columns=_LATENCY_CURVE_COLUMNS)
    return _ORIGINAL_LATENCY_CURVE(
        estimates_by_latency,
        truth,
        max_time_delta_s=normalized_gate,
    )


def domain_shift_summary(
    training: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame] | pd.DataFrame,
    heldout: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare finite held-out distributions against finite training data."""

    train = _LEGACY._concat_training_frames(training)
    if columns is None:
        columns = [
            column
            for column in heldout.columns
            if column in train.columns
            and pd.api.types.is_numeric_dtype(heldout[column])
        ]

    rows: list[dict[str, object]] = []
    for column in columns:
        train_values = pd.to_numeric(
            train[column], errors="coerce"
        ).to_numpy(dtype=float)
        heldout_values = pd.to_numeric(
            heldout[column], errors="coerce"
        ).to_numpy(dtype=float)
        train_values = train_values[np.isfinite(train_values)]
        heldout_values = heldout_values[np.isfinite(heldout_values)]
        if train_values.size == 0 or heldout_values.size == 0:
            continue

        train_std = float(np.std(train_values))
        if train_std == 0.0:
            train_std = 1.0
        rows.append(
            {
                "feature": column,
                "train_count": int(train_values.size),
                "heldout_count": int(heldout_values.size),
                "train_mean": float(np.mean(train_values)),
                "heldout_mean": float(np.mean(heldout_values)),
                "mean_shift_z": float(
                    (np.mean(heldout_values) - np.mean(train_values)) / train_std
                ),
                "train_p50": float(np.percentile(train_values, 50)),
                "heldout_p50": float(np.percentile(heldout_values, 50)),
                "train_p90": float(np.percentile(train_values, 90)),
                "heldout_p90": float(np.percentile(heldout_values, 90)),
                "ks_distance": _LEGACY._ks_distance(
                    train_values, heldout_values
                ),
            }
        )

    result = pd.DataFrame.from_records(rows, columns=_DOMAIN_SHIFT_COLUMNS)
    if result.empty:
        return result
    return result.sort_values(
        ["ks_distance", "feature"], ascending=[False, True]
    ).reset_index(drop=True)


_LEGACY._optional_float = _optional_float
_LEGACY._optional_int = _optional_int
_LEGACY._radar_frame_groups = _radar_frame_groups
_LEGACY._radar_event_key = _radar_event_key
_LEGACY._row_event_key = _row_event_key
_LEGACY._nearest_truth_position = _nearest_truth_position
_LEGACY.latency_curve = latency_curve
_LEGACY.track_switch_metrics = track_switch_metrics
_LEGACY.domain_shift_summary = domain_shift_summary

for _name in dir(_LEGACY):
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = getattr(_LEGACY, _name)
globals()["_event_index_value"] = _event_index_value
globals()["_radar_frame_groups"] = _radar_frame_groups
globals()["_radar_event_key"] = _radar_event_key
globals()["_row_event_key"] = _row_event_key
globals()["_nearest_truth_position"] = _nearest_truth_position
globals()["latency_curve"] = latency_curve
globals()["track_switch_metrics"] = track_switch_metrics
globals()["domain_shift_summary"] = domain_shift_summary

__doc__ = _LEGACY.__doc__
__all__ = [
    name
    for name in dir(_LEGACY)
    if not (name.startswith("__") and name.endswith("__"))
]
