"""Compatibility fixes for oracle-coverage identifiers and numeric controls.

The maintained implementation lives in the sibling ``oracle_coverage.py``
module. This package preserves the public import path while preventing
fractional identifiers from being truncated, large exact identifiers from being
rounded through binary floating point, malformed radar standard deviations from
silently changing candidate scoring, malformed explicit tracklet configurations
from disappearing behind defaults, serialized Boolean diagnostics from
corrupting oracle-retention summaries, equal class-probability scores from
receiving row-order-dependent ranks, and distinct candidate rows from collapsing
onto one diagnostic identity.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _shared_optional_float
from raft_uav.numeric import optional_int as _shared_optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "oracle_coverage.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._oracle_coverage_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load oracle coverage implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_ORACLE_CANDIDATE_COVERAGE = _IMPL.build_oracle_candidate_coverage
_ORIGINAL_ORACLE_COVERAGE_ROW = _IMPL._oracle_coverage_row
_ORIGINAL_COVERAGE_SUMMARY = _IMPL._coverage_summary
_ORIGINAL_BUCKET_SUMMARY = _IMPL._bucket_summary
_IDENTIFIER_KEY_COLUMNS = frozenset({"frame_index", "track_index", "track_id"})
_CANDIDATE_KEY_COLUMNS = (
    "frame_index",
    "track_index",
    "track_id",
    "time_s",
    "east_m",
    "north_m",
    "up_m",
)
_CANDIDATE_ROW_POSITION_KEY = "__candidate_row_position__"
_TRUE_BOOLEAN_TEXT = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE_BOOLEAN_TEXT = frozenset(
    {"false", "f", "no", "n", "0", "0.0", "", "nan", "none", "<na>", "nat"}
)


def _positive_standard_deviation(name: str, value: object) -> float:
    """Return a finite positive real scalar standard deviation."""

    standard_deviation = _shared_optional_float(value)
    if standard_deviation is None or standard_deviation <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return standard_deviation


def _validated_tracklet_config(config: object | None) -> Any:
    """Return a valid explicit tracklet configuration or the default instance."""

    if config is None:
        return _IMPL.TrackletViterbiAssociationConfig()
    if not isinstance(config, _IMPL.TrackletViterbiAssociationConfig):
        raise ValueError(
            "config must be a TrackletViterbiAssociationConfig instance or None"
        )
    return config


def _is_complex_scalar(value: Any) -> bool:
    """Return whether scalar array wrappers contain a complex value."""

    scalar = value
    seen_arrays: set[int] = set()
    while isinstance(scalar, np.ndarray):
        if scalar.ndim != 0:
            return False
        array_id = id(scalar)
        if array_id in seen_arrays:
            return False
        seen_arrays.add(array_id)
        scalar = scalar.item()
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    return isinstance(scalar, complex)


def _serialized_boolean_series(values: Any, *, column: str) -> pd.Series:
    """Parse native and serialized Boolean diagnostics without lossy coercion."""

    series = pd.Series(values)
    if series.empty:
        return pd.Series(index=series.index, dtype=bool)

    complex_values = series.map(_is_complex_scalar).fillna(False).astype(bool)
    if bool(complex_values.any()):
        invalid_indices = complex_values[complex_values].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        raise ValueError(
            f"{column} contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )

    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    missing = series.isna() | text.isin(_FALSE_BOOLEAN_TEXT)
    truthy = (text.isin(_TRUE_BOOLEAN_TEXT) | numeric.eq(1.0)).fillna(False)
    falsy = (missing | numeric.eq(0.0)).fillna(False)
    invalid = ~(truthy | falsy)
    if bool(invalid.any()):
        invalid_indices = invalid[invalid].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        raise ValueError(
            f"{column} contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return truthy.astype(bool)


def _normalized_summary_frame(frame: Any) -> Any:
    """Return a copy with summary Boolean diagnostics normalized explicitly."""

    normalized = frame.copy()
    for column in ("oracle_available", "oracle_retained"):
        if column in normalized.columns:
            normalized[column] = _serialized_boolean_series(
                normalized[column],
                column=column,
            )
    return normalized


def _coverage_summary(
    frame: Any,
    *,
    candidate_catprob_threshold: float | None,
    config: Any,
    truth_time_gate_s: float | None,
) -> dict[str, Any]:
    """Summarize oracle coverage after deterministic Boolean normalization."""

    return _ORIGINAL_COVERAGE_SUMMARY(
        _normalized_summary_frame(frame),
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
        truth_time_gate_s=truth_time_gate_s,
    )


def _bucket_summary(frame: Any) -> Any:
    """Build oracle-coverage buckets after deterministic Boolean normalization."""

    return _ORIGINAL_BUCKET_SUMMARY(_normalized_summary_frame(frame))


def build_oracle_candidate_coverage(
    *,
    radar: Any,
    truth: Any,
    rf_measurements: Any = (),
    candidate_catprob_threshold: float | None = 0.5,
    config: Any = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: object = 25.0,
    radar_z_std_m: object = 35.0,
    truth_time_gate_s: float | None = 1.0,
    gate_probabilities_by_source: Any = None,
    gate_thresholds_by_source: Any = None,
    safety_gate_probabilities_by_source: Any = None,
    safety_gate_thresholds_by_source: Any = None,
    robust_update_by_source: Any = None,
    inflation_alpha_by_source: Any = None,
    max_residual_norms_by_source: Any = None,
) -> Any:
    """Build oracle coverage after validating public configuration controls."""

    validated_config = _validated_tracklet_config(config)
    validated_xy_std_m = _positive_standard_deviation(
        "radar_xy_std_m",
        radar_xy_std_m,
    )
    validated_z_std_m = _positive_standard_deviation(
        "radar_z_std_m",
        radar_z_std_m,
    )
    return _ORIGINAL_BUILD_ORACLE_CANDIDATE_COVERAGE(
        radar=radar,
        truth=truth,
        rf_measurements=rf_measurements,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=validated_config,
        acceleration_std_mps2=acceleration_std_mps2,
        radar_xy_std_m=validated_xy_std_m,
        radar_z_std_m=validated_z_std_m,
        truth_time_gate_s=truth_time_gate_s,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )


def _optional_int(value: object) -> int | None:
    """Return an exact integer-equivalent scalar without truncation or rounding."""

    return _shared_optional_int(value)


def _candidate_key(row: Any) -> tuple[tuple[str, object], ...]:
    """Build an exact frame-local candidate key."""

    row_position = _optional_int(row.name)
    stable_position: object = row_position if row_position is not None else str(row.name)
    key: list[tuple[str, object]] = [
        (_CANDIDATE_ROW_POSITION_KEY, stable_position)
    ]
    columns = [column for column in _CANDIDATE_KEY_COLUMNS if column in row.index]
    for column in columns:
        value = row[column]
        if column in _IDENTIFIER_KEY_COLUMNS:
            exact = _optional_int(value)
            stable = exact if exact is not None else str(value)
        else:
            stable = _IMPL._stable_value(value)
        key.append((column, stable))
    return tuple(key)


def _oracle_coverage_row(
    *,
    candidates: Any,
    **kwargs: Any,
) -> Any:
    """Evaluate one frame after assigning unique frame-local row positions."""

    return _ORIGINAL_ORACLE_COVERAGE_ROW(
        candidates=candidates.reset_index(drop=True),
        **kwargs,
    )


def _event_key(candidates: Any, time_s: float) -> str:
    """Report the first valid frame identifier without a float round-trip."""

    if "frame_index" in candidates.columns and not candidates.empty:
        for value in candidates["frame_index"].tolist():
            frame_index = _optional_int(value)
            if frame_index is not None:
                return f"frame_index:{frame_index}"
    return f"time_s:{float(time_s):.9f}"


def _rank_by_catprob(candidates: Any, oracle_iloc: int) -> float:
    """Return a descending average rank so equal probabilities remain tied."""

    if "cat_prob_uav" not in candidates.columns:
        return float("nan")
    position = int(oracle_iloc)
    if position < 0 or position >= len(candidates):
        return float("nan")
    values = pd.to_numeric(
        candidates["cat_prob_uav"],
        errors="coerce",
    ).fillna(float("-inf"))
    ranks = values.rank(method="average", ascending=False)
    return float(ranks.iloc[position])


_IMPL.build_oracle_candidate_coverage = build_oracle_candidate_coverage
_IMPL._coverage_summary = _coverage_summary
_IMPL._bucket_summary = _bucket_summary
_IMPL._optional_int = _optional_int
_IMPL._candidate_key = _candidate_key
_IMPL._oracle_coverage_row = _oracle_coverage_row
_IMPL._event_key = _event_key
_IMPL._rank_by_catprob = _rank_by_catprob

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_positive_standard_deviation"] = _positive_standard_deviation
globals()["_validated_tracklet_config"] = _validated_tracklet_config
globals()["_is_complex_scalar"] = _is_complex_scalar
globals()["_serialized_boolean_series"] = _serialized_boolean_series
globals()["_normalized_summary_frame"] = _normalized_summary_frame
globals()["_coverage_summary"] = _coverage_summary
globals()["_bucket_summary"] = _bucket_summary
globals()["build_oracle_candidate_coverage"] = build_oracle_candidate_coverage
globals()["_optional_int"] = _optional_int
globals()["_candidate_key"] = _candidate_key
globals()["_oracle_coverage_row"] = _oracle_coverage_row
globals()["_event_key"] = _event_key
globals()["_rank_by_catprob"] = _rank_by_catprob

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
