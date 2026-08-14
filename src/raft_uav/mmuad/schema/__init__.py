"""Compatibility fixes for MMUAD schema normalization.

The maintained implementation lives in the sibling ``schema.py`` module. This
package preserves the public import path while ensuring that values nested in
NumPy arrays are normalized recursively before JSON serialization, that the
``NaT`` missing-value sentinel is not retained as a literal sequence identifier,
that timestamp aliases reject lossy Boolean and complex pseudo-scalars, and that
case- or whitespace-equivalent column names cannot be selected ambiguously.
"""

from __future__ import annotations

from collections.abc import Iterable
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_float as _optional_float

_IMPL_PATH = Path(__file__).resolve().parent.parent / "schema.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._schema_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load MMUAD schema implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_LOAD_JSONABLE = _IMPL.load_jsonable
_ORIGINAL_NORMALIZE_SEQUENCE_ID_VALUES = _IMPL._normalize_sequence_id_values


def load_jsonable(value: Any) -> Any:
    """Return recursively normalized JSON-safe values, including array elements."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    return _ORIGINAL_LOAD_JSONABLE(value)


def _normalize_sequence_id_values(
    values: pd.Series,
    *,
    default_sequence_id: str,
) -> pd.Series:
    """Normalize sequence ids, including serialized ``NaT`` sentinels."""

    normalized = _ORIGINAL_NORMALIZE_SEQUENCE_ID_VALUES(
        values,
        default_sequence_id=default_sequence_id,
    )
    missing_nat = normalized.astype(str).str.strip().str.casefold().eq("nat")
    return normalized.where(~missing_nat, str(default_sequence_id))


def _column_lookup(columns: Iterable[object]) -> dict[str, object]:
    """Return normalized column lookup after rejecting ambiguous names."""

    lookup: dict[str, object] = {}
    for column in columns:
        key = _IMPL._column_key(column)
        if key in lookup:
            raise ValueError(
                "column names are ambiguous after trimming whitespace and ignoring case: "
                f"{lookup[key]!r} and {column!r} both normalize to {key!r}"
            )
        lookup[key] = column
    return lookup


def _stamp_mapping_to_seconds(value: Any, seen_mapping_ids: set[int]) -> float | None:
    """Parse one ROS-style stamp mapping without lossy scalar coercion."""

    mapping = _IMPL._coerce_stamp_mapping(value)
    if mapping is None:
        return None
    mapping_id = id(mapping)
    if mapping_id in seen_mapping_ids:
        return None
    seen_mapping_ids.add(mapping_id)
    try:
        nested = _IMPL._mapping_get_case_insensitive(mapping, "stamp")
        if nested is not None:
            nested_time = _stamp_mapping_to_seconds(nested, seen_mapping_ids)
            if nested_time is not None:
                return nested_time

        seconds = _IMPL._first_mapping_value_case_insensitive(
            mapping,
            ("sec", "secs", "seconds"),
        )
        nanoseconds = _IMPL._first_mapping_value_case_insensitive(
            mapping,
            ("nanosec", "nsec", "nsecs", "nanoseconds"),
        )
        if seconds is not None:
            seconds_value = _optional_float(seconds)
            if seconds_value is None:
                return None
            if _IMPL._is_json_missing_scalar(nanoseconds):
                nanoseconds_value = 0.0
            else:
                nanoseconds_value = _optional_float(nanoseconds)
                if nanoseconds_value is None:
                    return None
            return _optional_float(seconds_value + nanoseconds_value * 1.0e-9)

        for alias, scale in _IMPL._TIME_UNIT_ALIASES.items():
            scalar = _IMPL._mapping_get_case_insensitive(mapping, alias)
            if scalar is None:
                continue
            scalar_value = _optional_float(scalar)
            if scalar_value is not None:
                return _optional_float(scalar_value * scale)

        for alias in ("time_s", "timestamp_s", "timestamp", "stamp", "time"):
            scalar = _IMPL._mapping_get_case_insensitive(mapping, alias)
            if scalar is None:
                continue
            scalar_value = _optional_float(scalar)
            if scalar_value is not None:
                return scalar_value
        return None
    finally:
        seen_mapping_ids.remove(mapping_id)


def _stamp_dict_to_seconds(value: Any) -> float | None:
    """Return a finite real timestamp from a ROS-style stamp mapping."""

    return _stamp_mapping_to_seconds(value, set())


def _finite_time_series(values: pd.Series) -> pd.Series:
    """Normalize finite real timestamp scalars and mark malformed cells missing."""

    normalized = [_optional_float(value) for value in values.tolist()]
    return pd.Series(
        [np.nan if value is None else value for value in normalized],
        index=values.index,
        dtype=float,
    )


def _stamp_pair_series(seconds: pd.Series, nanoseconds: pd.Series) -> pd.Series:
    """Combine ROS seconds/nanoseconds columns without Boolean coercion."""

    normalized: list[float] = []
    for seconds_value, nanoseconds_value in zip(
        seconds.tolist(),
        nanoseconds.tolist(),
        strict=True,
    ):
        parsed_seconds = _optional_float(seconds_value)
        if parsed_seconds is None:
            normalized.append(np.nan)
            continue
        if _IMPL._is_json_missing_scalar(nanoseconds_value):
            parsed_nanoseconds = 0.0
        else:
            parsed_nanoseconds = _optional_float(nanoseconds_value)
            if parsed_nanoseconds is None:
                normalized.append(np.nan)
                continue
        timestamp = _optional_float(parsed_seconds + parsed_nanoseconds * 1.0e-9)
        normalized.append(np.nan if timestamp is None else timestamp)
    return pd.Series(normalized, index=seconds.index, dtype=float)


def _seconds_or_stamp_dict_series(values: pd.Series) -> pd.Series:
    """Return finite real seconds from scalars or ROS stamp dictionaries."""

    normalized: list[float] = []
    for value in values.tolist():
        timestamp = _optional_float(value)
        if timestamp is None:
            timestamp = _stamp_dict_to_seconds(value)
        normalized.append(np.nan if timestamp is None else timestamp)
    return pd.Series(normalized, index=values.index, dtype=float)


def _time_alias_series(
    frame: pd.DataFrame,
    lower_to_original: dict[str, object],
) -> pd.Series | None:
    """Build timestamp aliases while rejecting lossy pseudo-scalars row-wise."""

    candidates: list[pd.Series] = []
    for seconds_alias, nanoseconds_alias in _IMPL._TIME_SECOND_NANOSECOND_PAIRS:
        seconds_col = lower_to_original.get(_IMPL._column_key(seconds_alias))
        nanoseconds_col = lower_to_original.get(_IMPL._column_key(nanoseconds_alias))
        if seconds_col is None or nanoseconds_col is None:
            continue
        candidates.append(
            _stamp_pair_series(frame[seconds_col], frame[nanoseconds_col])
        )
    for alias, scale in _IMPL._TIME_UNIT_ALIASES.items():
        original = lower_to_original.get(_IMPL._column_key(alias))
        if original is not None:
            candidates.append(_finite_time_series(frame[original]) * scale)
    for alias in _IMPL._TIME_SECOND_ALIASES:
        original = lower_to_original.get(_IMPL._column_key(alias))
        if original is not None and _IMPL._column_key(original) != "time_s":
            candidates.append(_seconds_or_stamp_dict_series(frame[original]))
    for alias in ("header.stamp", "header"):
        original = lower_to_original.get(_IMPL._column_key(alias))
        if original is not None:
            candidates.append(_seconds_or_stamp_dict_series(frame[original]))
    combined = _IMPL._combine_time_alias_series(candidates)
    if combined is None and candidates:
        return candidates[0]
    return combined


_IMPL.load_jsonable = load_jsonable
_IMPL._normalize_sequence_id_values = _normalize_sequence_id_values
_IMPL._column_lookup = _column_lookup
_IMPL._stamp_dict_to_seconds = _stamp_dict_to_seconds
_IMPL._seconds_or_stamp_dict_series = _seconds_or_stamp_dict_series
_IMPL._time_alias_series = _time_alias_series

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["load_jsonable"] = load_jsonable
globals()["_normalize_sequence_id_values"] = _normalize_sequence_id_values
globals()["_column_lookup"] = _column_lookup
globals()["_stamp_dict_to_seconds"] = _stamp_dict_to_seconds
globals()["_seconds_or_stamp_dict_series"] = _seconds_or_stamp_dict_series
globals()["_time_alias_series"] = _time_alias_series

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]