"""Compatibility fixes for Track 5 classification relabel validation.

The maintained implementation lives in the sibling
``track5_classification_relabel.py`` module. This package preserves the public
import path while requiring exact integer class labels, genuine sequence
identifiers, unique official row keys, valid nearest-time gates, and valid
sequence-class probability mass before relabeling.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_classification_relabel.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_classification_relabel_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 classification relabel implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

VALID_CLASS_IDS = _IMPL.VALID_CLASS_IDS
_ORIGINAL_NORMALIZE_FRAME = _IMPL._normalize_frame
_ORIGINAL_NEAREST_TIME_RELABEL_MERGE = _IMPL._nearest_time_relabel_merge
_ORIGINAL_SEQUENCE_PREDICTION_LABELS = _IMPL._sequence_prediction_labels


def _reject_boolean_class_labels(values: Any, *, name: str) -> None:
    """Reject Boolean pseudo-numbers before pandas can coerce them to class IDs."""

    raw = pd.Series(values, copy=False)
    boolean = raw.map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean.any():
        raise ValueError(f"{name} contains Boolean class labels")


def _validate_class_series(values: pd.Series, *, name: str) -> None:
    """Require finite labels exactly equal to official integer class IDs."""

    raw = pd.Series(values, copy=False)
    _reject_boolean_class_labels(raw, name=name)

    numeric = pd.to_numeric(raw, errors="coerce")
    numeric_values = numeric.to_numpy(float)
    if numeric.isna().any() or not np.isfinite(numeric_values).all():
        raise ValueError(f"{name} contains non-finite class labels")

    rounded_values = np.rint(numeric_values)
    if not np.equal(numeric_values, rounded_values).all():
        raise ValueError(f"{name} contains non-integer class labels")

    integer_values = pd.Series(
        rounded_values.astype(int),
        index=numeric.index,
    )
    bad = sorted(
        set(
            integer_values.loc[~integer_values.isin(VALID_CLASS_IDS)]
            .astype(int)
            .tolist()
        )
    )
    if bad:
        allowed = ", ".join(str(class_id) for class_id in VALID_CLASS_IDS)
        raise ValueError(f"{name} contains class labels outside {{{allowed}}}: {bad}")


def _validate_sequence_ids(values: Any, *, name: str) -> None:
    """Reject genuinely missing or blank sequence identifiers before string conversion."""

    raw = pd.Series(values, copy=False)
    missing = raw.isna()
    text = raw.where(~missing, "").astype(str).str.strip()
    invalid = missing | text.eq("")
    if invalid.any():
        rows = invalid.index[invalid].tolist()[:5]
        raise ValueError(
            f"{name} contains missing or blank Sequence identifiers at rows {rows}"
        )


def _validate_unique_row_keys(rows: pd.DataFrame, *, name: str) -> None:
    """Reject duplicate official ``Sequence``/``Timestamp`` row keys."""

    duplicate = rows.duplicated(subset=["Sequence", "Timestamp"], keep=False)
    if not duplicate.any():
        return
    keys = rows.loc[duplicate, ["Sequence", "Timestamp"]].drop_duplicates().head(5)
    examples = [
        (str(row.Sequence), float(row.Timestamp))
        for row in keys.itertuples(index=False)
    ]
    raise ValueError(
        f"{name} contains duplicate Sequence/Timestamp keys: {examples}"
    )


def _normalize_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Validate official sequence identifiers and row keys before relabeling."""

    rows = pd.DataFrame(frame).copy()
    if "Sequence" in rows.columns:
        _validate_sequence_ids(rows["Sequence"], name=name)
    if "Classification" in rows.columns:
        _reject_boolean_class_labels(
            rows["Classification"],
            name=f"{name}.Classification",
        )
    normalized = _ORIGINAL_NORMALIZE_FRAME(rows, name=name)
    _validate_unique_row_keys(normalized, name=name)
    return normalized


def _normalize_optional_nonnegative_float(value: Any, *, field: str) -> float | None:
    """Return an optional finite non-negative scalar with a stable error."""

    if value is None:
        return None
    message = f"{field} must be a finite non-negative number"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0:
        raise ValueError(message)
    item = scalar.item()
    if isinstance(item, (bool, np.bool_)) or np.ma.is_masked(item) or np.iscomplexobj(item):
        raise ValueError(message)
    try:
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(message)
    return number


def _nearest_time_relabel_merge(
    pose: pd.DataFrame,
    source: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None,
) -> pd.DataFrame:
    """Validate nearest-time tolerance before matching classification rows."""

    tolerance = _normalize_optional_nonnegative_float(
        max_nearest_time_delta_s,
        field="max_nearest_time_delta_s",
    )
    return _ORIGINAL_NEAREST_TIME_RELABEL_MERGE(
        pose,
        source,
        max_nearest_time_delta_s=tolerance,
    )


def _validate_sequence_probability_rows(
    rows: pd.DataFrame,
    *,
    sequence_column: Any,
    probability_items: list[tuple[int, Any]],
) -> None:
    """Reject malformed probability cells and sequence groups with zero mass."""

    probability_columns = [column for _class_id, column in probability_items]
    numeric = pd.DataFrame(index=rows.index)
    invalid_examples: list[str] = []
    for column in probability_columns:
        converted = pd.to_numeric(rows[column], errors="coerce")
        values = converted.to_numpy(float)
        invalid = ~np.isfinite(values) | (values < 0.0)
        if invalid.any() and len(invalid_examples) < 5:
            for position in np.flatnonzero(invalid):
                row_index = rows.index[int(position)]
                invalid_examples.append(
                    f"row {row_index}, {column!r}={rows.iloc[int(position)][column]!r}"
                )
                if len(invalid_examples) >= 5:
                    break
        numeric[column] = converted

    if invalid_examples:
        raise ValueError(
            "sequence prediction probabilities contain non-finite, non-numeric, "
            f"or negative values: {invalid_examples}"
        )

    sequence_ids = rows[sequence_column].astype(str).str.strip()
    grouped = numeric.assign(_Sequence=sequence_ids).groupby("_Sequence", sort=True).mean()
    totals = grouped.to_numpy(float).sum(axis=1)
    empty = grouped.index[totals <= 0.0].astype(str).tolist()
    if empty:
        preview = ", ".join(repr(sequence_id) for sequence_id in empty[:10])
        suffix = "" if len(empty) <= 10 else f", ... ({len(empty)} total)"
        raise ValueError(
            "sequence prediction probabilities have no positive mass for "
            f"sequence(s): {preview}{suffix}"
        )


def _sequence_prediction_labels(sequence_predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate aliased prediction identifiers and values before aggregation."""

    rows = pd.DataFrame(sequence_predictions).copy()
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    if sequence_column is not None:
        _validate_sequence_ids(
            rows[sequence_column],
            name="sequence prediction table",
        )
    class_column = _IMPL._first_present(rows, _IMPL.PREDICTED_CLASS_ALIASES)
    if class_column is not None:
        _reject_boolean_class_labels(
            rows[class_column],
            name=f"sequence prediction table.{class_column}",
        )
    probability_items = _IMPL._probability_columns(rows)
    probability_class_ids = tuple(class_id for class_id, _column in probability_items)
    if (
        sequence_column is not None
        and _IMPL._valid_probability_class_ids(probability_class_ids)
    ):
        _validate_sequence_probability_rows(
            rows,
            sequence_column=sequence_column,
            probability_items=probability_items,
        )
    return _ORIGINAL_SEQUENCE_PREDICTION_LABELS(rows)


_IMPL._reject_boolean_class_labels = _reject_boolean_class_labels
_IMPL._validate_class_series = _validate_class_series
_IMPL._validate_sequence_ids = _validate_sequence_ids
_IMPL._validate_unique_row_keys = _validate_unique_row_keys
_IMPL._normalize_optional_nonnegative_float = _normalize_optional_nonnegative_float
_IMPL._normalize_frame = _normalize_frame
_IMPL._nearest_time_relabel_merge = _nearest_time_relabel_merge
_IMPL._validate_sequence_probability_rows = _validate_sequence_probability_rows
_IMPL._sequence_prediction_labels = _sequence_prediction_labels

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_reject_boolean_class_labels"] = _reject_boolean_class_labels
globals()["_validate_class_series"] = _validate_class_series
globals()["_validate_sequence_ids"] = _validate_sequence_ids
globals()["_validate_unique_row_keys"] = _validate_unique_row_keys
globals()["_normalize_optional_nonnegative_float"] = _normalize_optional_nonnegative_float
globals()["_normalize_frame"] = _normalize_frame
globals()["_nearest_time_relabel_merge"] = _nearest_time_relabel_merge
globals()["_validate_sequence_probability_rows"] = _validate_sequence_probability_rows
globals()["_sequence_prediction_labels"] = _sequence_prediction_labels

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
