"""Scope bias-calibration residual matching by physical flight metadata."""

from __future__ import annotations

import importlib
from typing import Sequence

import numpy as np
import pandas as pd

_bias = importlib.import_module("raft_uav.calibration.bias")
_ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES = _bias.make_bias_training_examples
_SCOPE_COLUMNS = ("sequence_id", "flight_id")
_MISSING_SCOPE_VALUES = frozenset({"nan", "none", "<na>", "nat"})


def _normalized_scope_value(value: object) -> str | None:
    """Return a stable physical-scope identifier, or ``None`` when missing."""

    if value is None or value is pd.NA or np.ma.is_masked(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    text = str(value).strip()
    if not text or text.casefold() in _MISSING_SCOPE_VALUES:
        return None
    return text


def _available_scope_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return physical-scope columns carried by ``frame``."""

    return tuple(column for column in _SCOPE_COLUMNS if column in frame.columns)


def _column_has_scope_value(frame: pd.DataFrame, column: str) -> bool:
    """Return whether ``column`` contains at least one usable scope identifier."""

    if column not in frame.columns or frame.empty:
        return False
    return bool(frame[column].map(_normalized_scope_value).notna().any())


def _scope_keys(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> list[tuple[str | None, ...]]:
    """Return normalized joint scope keys in input-row order."""

    if not columns:
        return [tuple()] * len(frame)
    normalized = [
        frame[column].map(_normalized_scope_value).tolist()
        for column in columns
    ]
    return [
        tuple(values)
        for values in zip(*normalized, strict=True)
    ]


def _extra_scope_subdivides_shared_scope(
    frame: pd.DataFrame,
    *,
    shared_columns: tuple[str, ...],
    extra_columns: tuple[str, ...],
) -> bool:
    """Return whether one-sided metadata distinguishes rows inside a shared scope."""

    if frame.empty or not extra_columns:
        return False
    shared_keys = _scope_keys(frame, shared_columns)
    extra_keys = _scope_keys(frame, extra_columns)
    extras_by_shared: dict[
        tuple[str | None, ...],
        set[tuple[str | None, ...]],
    ] = {}
    for shared_key, extra_key in zip(shared_keys, extra_keys, strict=True):
        extras_by_shared.setdefault(shared_key, set()).add(extra_key)
    return any(len(extra_keys_for_scope) > 1 for extra_keys_for_scope in extras_by_shared.values())


def _scope_cardinality(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> int:
    """Return the number of distinct normalized scopes represented by ``frame``."""

    if frame.empty or not columns:
        return 0
    return len(set(_scope_keys(frame, columns)))


def _temporary_column(frame: pd.DataFrame, stem: str) -> str:
    """Return a private column name that does not overwrite caller data."""

    candidate = stem
    suffix = 0
    while candidate in frame.columns:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    return candidate


def _scoped_bias_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    scope_columns: tuple[str, ...],
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float,
) -> pd.DataFrame:
    """Run the established bias matcher independently inside each physical scope."""

    order_column = _temporary_column(measurements, "__raft_uav_bias_input_order")
    ordered_measurements = measurements.copy()
    ordered_measurements[order_column] = np.arange(len(measurements), dtype=np.int64)
    measurement_keys = _scope_keys(ordered_measurements, scope_columns)
    truth_keys = _scope_keys(truth, scope_columns)

    pieces: list[pd.DataFrame] = []
    for scope_key in dict.fromkeys(measurement_keys):
        measurement_mask = np.fromiter(
            (key == scope_key for key in measurement_keys),
            dtype=bool,
            count=len(measurement_keys),
        )
        truth_mask = np.fromiter(
            (key == scope_key for key in truth_keys),
            dtype=bool,
            count=len(truth_keys),
        )
        if not bool(np.any(truth_mask)):
            continue
        rows = _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
            ordered_measurements.loc[measurement_mask].reset_index(drop=True),
            truth.loc[truth_mask].reset_index(drop=True),
            source=source,
            target_columns=target_columns,
            time_gate_s=time_gate_s,
        )
        if not rows.empty:
            pieces.append(rows)

    if not pieces:
        return pd.DataFrame()
    return (
        pd.concat(pieces, ignore_index=True, sort=False)
        .sort_values(order_column, kind="stable")
        .drop(columns=order_column)
        .reset_index(drop=True)
    )


def make_bias_training_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float = 2.0,
) -> pd.DataFrame:
    """Build bias examples without matching observations across physical flights."""

    measurement_columns = _available_scope_columns(measurements)
    truth_columns = _available_scope_columns(truth)
    common_columns = tuple(
        column
        for column in _SCOPE_COLUMNS
        if column in measurement_columns and column in truth_columns
    )
    shared_columns = tuple(
        column
        for column in common_columns
        if _column_has_scope_value(measurements, column)
        or _column_has_scope_value(truth, column)
    )

    measurement_extra = tuple(
        column for column in measurement_columns if column not in common_columns
    )
    truth_extra = tuple(
        column for column in truth_columns if column not in common_columns
    )
    if _extra_scope_subdivides_shared_scope(
        measurements,
        shared_columns=shared_columns,
        extra_columns=measurement_extra,
    ) or _extra_scope_subdivides_shared_scope(
        truth,
        shared_columns=shared_columns,
        extra_columns=truth_extra,
    ):
        raise ValueError(
            "pooled bias calibration requires all disambiguating sequence_id and "
            "flight_id metadata on both measurements and truth"
        )

    if not shared_columns:
        measurement_scope_count = _scope_cardinality(
            measurements,
            measurement_columns,
        )
        truth_scope_count = _scope_cardinality(truth, truth_columns)
        if measurement_scope_count > 1 or truth_scope_count > 1:
            raise ValueError(
                "pooled bias calibration requires a shared sequence_id or flight_id"
            )
        return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
            measurements,
            truth,
            source=source,
            target_columns=target_columns,
            time_gate_s=time_gate_s,
        )

    measurement_keys = _scope_keys(measurements, shared_columns)
    truth_keys = _scope_keys(truth, shared_columns)
    pooled = len(set((*measurement_keys, *truth_keys))) > 1
    if not pooled:
        return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
            measurements,
            truth,
            source=source,
            target_columns=target_columns,
            time_gate_s=time_gate_s,
        )

    if any(any(value is None for value in key) for key in measurement_keys):
        raise ValueError(
            "pooled bias calibration requires complete physical-scope metadata "
            "on every measurement row"
        )
    if any(any(value is None for value in key) for key in truth_keys):
        raise ValueError(
            "pooled bias calibration requires complete physical-scope metadata "
            "on every truth row"
        )

    return _scoped_bias_examples(
        measurements,
        truth,
        scope_columns=shared_columns,
        source=source,
        target_columns=target_columns,
        time_gate_s=time_gate_s,
    )


_bias.make_bias_training_examples = make_bias_training_examples
_bias._IMPL.make_bias_training_examples = make_bias_training_examples
