"""Compatibility wrapper for strict in-memory class-probability inputs.

The maintained implementation lives in the sibling
``class_probability_context.py`` module. This package preserves the public
import path while applying the same header and sequence-id normalization used
by the shared CSV reader.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "class_probability_context.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._class_probability_context_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        "cannot load class-probability context implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_LEGACY_CANDIDATE_ROWS = _IMPL._candidate_rows
_LEGACY_PROBABILITY_ROWS = _IMPL._probability_rows
_MISSING_SEQUENCE_TEXT = frozenset({"nan", "none", "<na>"})


def _predicted_class_labels(values: pd.Series) -> pd.Series:
    """Return canonical labels only for exactly integer-equivalent values."""

    raw = pd.Series(values)
    text = raw.where(raw.notna(), "").astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    boolean_values = raw.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).to_numpy(dtype=bool)
    integer_like = (
        np.isfinite(numeric_array)
        & (numeric_array == np.rint(numeric_array))
        & ~boolean_values
    )
    if integer_like.any():
        positions = np.flatnonzero(integer_like)
        text.iloc[positions] = (
            np.rint(numeric_array[positions]).astype(int).astype(str)
        )
    return text


def _normalize_sequence_ids(
    values: pd.Series,
    *,
    table_name: str,
    column: str,
) -> pd.Series:
    """Return stripped sequence IDs while rejecting serialized missing values."""

    raw = pd.Series(values, index=values.index)
    text = raw.where(raw.notna(), "").astype(str).str.strip()
    missing = text.eq("") | text.str.casefold().isin(_MISSING_SEQUENCE_TEXT)
    if missing.any():
        row_position = int(np.flatnonzero(missing.to_numpy(dtype=bool))[0])
        bad_value = raw.iloc[row_position]
        raise ValueError(
            f"{table_name} sequence identifiers must be non-empty; "
            f"got {bad_value!r} in {column!r} at row {row_position}"
        )
    return text


def _candidate_rows(candidates: object) -> pd.DataFrame:
    """Normalize and validate candidate sequence identifiers before joining."""

    rows = _LEGACY_CANDIDATE_ROWS(candidates)
    if rows.empty:
        return rows
    rows = rows.copy()
    rows["sequence_id"] = _normalize_sequence_ids(
        rows["sequence_id"],
        table_name="candidate",
        column="sequence_id",
    )
    return rows


def _normalize_probability_input(class_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Normalize in-memory probability tables like the shared CSV reader."""

    rows = pd.DataFrame(class_probabilities).copy()
    normalized_columns = [str(column).strip() for column in rows.columns]
    columns_by_key: dict[str, list[str]] = {}
    for column in normalized_columns:
        columns_by_key.setdefault(column.casefold(), []).append(column)
    collisions = [
        columns
        for columns in columns_by_key.values()
        if len(columns) > 1
    ]
    if collisions:
        rendered = "; ".join(
            ", ".join(repr(column) for column in columns)
            for columns in collisions
        )
        raise ValueError(
            "class probability table has ambiguous columns after trimming "
            f"whitespace and ignoring case: {rendered}"
        )
    rows.columns = normalized_columns

    alias_keys = {str(alias).casefold() for alias in _IMPL.SEQUENCE_ALIASES}
    sequence_columns = [
        column
        for column in rows.columns
        if str(column).casefold() in alias_keys
    ]
    if not sequence_columns:
        return rows

    preferred_column = (
        "sequence_id" if "sequence_id" in sequence_columns else sequence_columns[0]
    )
    normalized_ids: dict[str, pd.Series] = {}
    for column in sequence_columns:
        normalized_ids[column] = _normalize_sequence_ids(
            rows[column],
            table_name="class probability",
            column=column,
        )

    preferred_ids = normalized_ids[preferred_column]
    for column, text in normalized_ids.items():
        if column == preferred_column:
            continue
        conflicts = text.ne(preferred_ids)
        if conflicts.any():
            row_index = int(np.flatnonzero(conflicts.to_numpy(dtype=bool))[0])
            raise ValueError(
                "class probability table has conflicting sequence identifier "
                f"columns {preferred_column!r} and {column!r} at row {row_index}"
            )
    for column, text in normalized_ids.items():
        rows[column] = text
    return rows


def _sanitize_probability_inputs(class_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Clean each probability entry before duplicate-sequence aggregation."""

    rows = pd.DataFrame(class_probabilities).copy()
    for label in _IMPL.OFFICIAL_CLASS_LABELS:
        source = _IMPL._probability_column(rows, label)
        if source is None:
            continue
        numeric = pd.to_numeric(rows[source], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        rows[source] = numeric.where(finite, 0.0).clip(lower=0.0)
    return rows


def _probability_rows(class_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Normalize probability rows after sanitizing their public inputs."""

    normalized = _normalize_probability_input(class_probabilities)
    sanitized = _sanitize_probability_inputs(normalized)
    return _LEGACY_PROBABILITY_ROWS(sanitized)


_IMPL._predicted_class_labels = _predicted_class_labels
_IMPL._candidate_rows = _candidate_rows
_IMPL._probability_rows = _probability_rows

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_predicted_class_labels"] = _predicted_class_labels
globals()["_normalize_sequence_ids"] = _normalize_sequence_ids
globals()["_candidate_rows"] = _candidate_rows
globals()["_normalize_probability_input"] = _normalize_probability_input
globals()["_sanitize_probability_inputs"] = _sanitize_probability_inputs
globals()["_probability_rows"] = _probability_rows

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
