"""Compatibility fixes for bias-correction calibration and application.

The maintained implementation lives in the sibling ``bias.py`` module. This
package preserves the public import path while ensuring ``correct_frame``
respects ``keep_uncorrected=False``, serialized truth timestamps are numeric
before nearest-time calibration sorting, duplicate truth timestamps retain
their final valid row within each sequence, pooled calibration stays
sequence-local, and genuinely complex calibration values are not silently
reduced to their real components.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "bias.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.calibration._bias_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load bias correction utilities from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES = _IMPL.make_bias_training_examples
_SEQUENCE_COLUMN_CANDIDATES = ("sequence_id", "flight_id")


def _correct_frame(
    self: object,
    frame: pd.DataFrame,
    *,
    keep_uncorrected: bool = True,
) -> pd.DataFrame:
    """Apply the model and optionally omit the retained raw target columns."""

    corrected = self.apply(frame)
    if keep_uncorrected:
        return corrected
    raw_columns = [f"raw_{column}" for column in self.target_columns]
    return corrected.drop(columns=raw_columns, errors="ignore")


def _finite_real_numeric_series(values: pd.Series) -> pd.Series:
    """Coerce numeric values without discarding nonzero imaginary components."""

    numeric = pd.to_numeric(values, errors="coerce")
    array = numeric.to_numpy()
    if np.iscomplexobj(array):
        real = np.real(array)
        imaginary = np.imag(array)
        valid = np.isfinite(real) & np.isfinite(imaginary) & (imaginary == 0.0)
        result = np.where(valid, real, np.nan)
    else:
        real = np.asarray(array, dtype=float)
        result = np.where(np.isfinite(real), real, np.nan)
    return pd.Series(result, index=values.index, dtype=float)


def _normalized_bias_frame(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Return a copy whose requested numeric columns contain finite real values."""

    normalized = frame.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = _finite_real_numeric_series(normalized[column])
    return normalized


def _drop_invalid_numeric_rows(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Drop rows invalidated while coercing required calibration columns."""

    if frame.empty:
        return frame.copy()
    required = tuple(str(column) for column in columns)
    if any(column not in frame.columns for column in required):
        return frame.copy()
    valid = np.isfinite(frame.loc[:, required].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[valid].reset_index(drop=True)


def _canonical_sequence_id(value: object) -> str | None:
    """Return a stable scalar sequence identifier, or ``None`` for missing values."""

    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    text = str(value).strip()
    return text or None


def _sequence_column(frame: pd.DataFrame) -> str | None:
    """Return the first populated supported sequence identifier column."""

    fallback: str | None = None
    for column in _SEQUENCE_COLUMN_CANDIDATES:
        if column not in frame.columns:
            continue
        if fallback is None:
            fallback = column
        if frame[column].map(_canonical_sequence_id).notna().any():
            return column
    return fallback


def _sequence_keys(frame: pd.DataFrame, column: str | None) -> pd.Series:
    """Return canonical sequence keys aligned with ``frame`` rows."""

    if column is None:
        return pd.Series([None] * len(frame), index=frame.index, dtype=object)
    return frame[column].map(_canonical_sequence_id).astype(object)


def _temporary_column(frame: pd.DataFrame, stem: str) -> str:
    """Return a private column name that cannot overwrite caller data."""

    candidate = stem
    suffix = 0
    while candidate in frame.columns:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    return candidate


def _keep_final_truth_rows(
    truth: pd.DataFrame,
    sequence_keys: pd.Series | None = None,
) -> pd.DataFrame:
    """Keep the final valid row for each timestamp, scoped by sequence when known."""

    if truth.empty or "time_s" not in truth.columns:
        return truth.copy()
    if sequence_keys is None:
        return truth.drop_duplicates(subset=["time_s"], keep="last").reset_index(
            drop=True
        )

    key_column = _temporary_column(truth, "__raft_uav_bias_sequence_key")
    keyed_truth = truth.copy()
    keyed_truth[key_column] = sequence_keys.to_numpy(dtype=object)
    return (
        keyed_truth.drop_duplicates(subset=[key_column, "time_s"], keep="last")
        .drop(columns=key_column)
        .reset_index(drop=True)
    )


def _sequence_local_bias_examples(
    measurements: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    measurement_keys: pd.Series,
    truth_keys: pd.Series,
    source: str,
    target_columns: Sequence[str],
    time_gate_s: float,
) -> pd.DataFrame:
    """Run nearest-time matching independently for each labeled sequence."""

    order_column = _temporary_column(measurements, "__raft_uav_bias_input_order")
    ordered_measurements = measurements.copy()
    ordered_measurements[order_column] = np.arange(len(measurements), dtype=np.int64)
    measurement_key_array = measurement_keys.to_numpy(dtype=object)
    truth_key_array = truth_keys.to_numpy(dtype=object)

    pieces: list[pd.DataFrame] = []
    for sequence_id in dict.fromkeys(measurement_key_array.tolist()):
        measurement_mask = measurement_key_array == sequence_id
        truth_mask = truth_key_array == sequence_id
        if not np.any(truth_mask):
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
    """Build bias examples from finite real, sequence-consistent inputs."""

    numeric_columns = ("time_s", *(str(column) for column in target_columns))
    normalized_measurements = _drop_invalid_numeric_rows(
        _normalized_bias_frame(measurements, numeric_columns),
        numeric_columns,
    )
    normalized_truth = _drop_invalid_numeric_rows(
        _normalized_bias_frame(truth, numeric_columns),
        numeric_columns,
    )

    required_columns = set(numeric_columns)
    has_required_columns = (
        required_columns.issubset(normalized_measurements.columns)
        and required_columns.issubset(normalized_truth.columns)
    )
    if (
        normalized_measurements.empty
        or normalized_truth.empty
        or not has_required_columns
    ):
        return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
            normalized_measurements,
            normalized_truth,
            source=source,
            target_columns=target_columns,
            time_gate_s=time_gate_s,
        )

    measurement_sequence_column = _sequence_column(normalized_measurements)
    truth_sequence_column = _sequence_column(normalized_truth)
    measurement_keys = _sequence_keys(
        normalized_measurements,
        measurement_sequence_column,
    )
    truth_keys = _sequence_keys(normalized_truth, truth_sequence_column)
    explicit_sequence_ids = {
        sequence_id for sequence_id in measurement_keys if sequence_id is not None
    }
    explicit_sequence_ids.update(
        sequence_id for sequence_id in truth_keys if sequence_id is not None
    )
    pooled = len(explicit_sequence_ids) > 1

    if pooled and (
        measurement_sequence_column is None or truth_sequence_column is None
    ):
        raise ValueError(
            "pooled bias calibration requires sequence_id or flight_id on both "
            "measurements and truth"
        )
    if pooled and (measurement_keys.isna().any() or truth_keys.isna().any()):
        raise ValueError(
            "pooled bias calibration requires a nonmissing sequence identifier "
            "on every measurement and truth row"
        )

    if pooled:
        normalized_truth = _keep_final_truth_rows(normalized_truth, truth_keys)
        truth_keys = _sequence_keys(normalized_truth, truth_sequence_column)
        return _sequence_local_bias_examples(
            normalized_measurements,
            normalized_truth,
            measurement_keys=measurement_keys,
            truth_keys=truth_keys,
            source=source,
            target_columns=target_columns,
            time_gate_s=time_gate_s,
        )

    normalized_truth = _keep_final_truth_rows(normalized_truth)
    return _ORIGINAL_MAKE_BIAS_TRAINING_EXAMPLES(
        normalized_measurements,
        normalized_truth,
        source=source,
        target_columns=target_columns,
        time_gate_s=time_gate_s,
    )


_IMPL.SensorBiasCorrectionModel.correct_frame = _correct_frame
_IMPL.make_bias_training_examples = make_bias_training_examples

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_correct_frame"] = _correct_frame
globals()["_finite_real_numeric_series"] = _finite_real_numeric_series
globals()["_normalized_bias_frame"] = _normalized_bias_frame
globals()["_drop_invalid_numeric_rows"] = _drop_invalid_numeric_rows
globals()["_canonical_sequence_id"] = _canonical_sequence_id
globals()["_sequence_column"] = _sequence_column
globals()["_sequence_keys"] = _sequence_keys
globals()["_temporary_column"] = _temporary_column
globals()["_keep_final_truth_rows"] = _keep_final_truth_rows
globals()["_sequence_local_bias_examples"] = _sequence_local_bias_examples
globals()["make_bias_training_examples"] = make_bias_training_examples

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
