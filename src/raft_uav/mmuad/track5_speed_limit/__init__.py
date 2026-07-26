"""Compatibility wrapper validating Track 5 speed-limit inputs.

The maintained implementation lives in the sibling ``track5_speed_limit.py``
module. This package preserves the public import path while rejecting malformed
iteration counts, Boolean pseudo-numbers, non-scalar controls, missing sequence
identifiers, invalid fixed-grid rows, invalid classification labels, and
duplicate fixed-grid keys instead of silently coercing or dropping them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.mmuad.submission import (
    normalize_official_track5_results_frame,
    parse_official_classification_cell,
)

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_speed_limit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_speed_limit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load speed-limit implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_PROJECT = _IMPL.project_track5_speed_limit
_NUMERIC_COLUMNS = (
    "time_s",
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "Classification",
)


def _finite_scalar(value: object, *, message: str) -> float:
    """Return one finite, non-Boolean scalar value."""

    if np.ma.is_masked(value):
        raise ValueError(message)
    scalar = value
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        scalar = value.item()
    elif isinstance(value, np.generic):
        scalar = value.item()
    if np.ma.is_masked(scalar) or isinstance(scalar, (bool, np.bool_)):
        raise ValueError(message)
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric):
        raise ValueError(message)
    return numeric


def _positive_integer(value: object, *, name: str) -> int:
    """Return a positive integer without lossy or Boolean coercion."""

    message = f"{name} must be a positive integer"
    numeric = _finite_scalar(value, message=message)
    if numeric <= 0.0 or not numeric.is_integer():
        raise ValueError(message)
    return int(numeric)


def _reject_boolean_scalar(value: object, *, message: str) -> float:
    """Normalize one finite scalar while rejecting Booleans and arrays."""

    return _finite_scalar(value, message=message)


def _validate_sequence_ids(submission: object) -> None:
    """Reject genuinely missing or blank sequence identifiers before string conversion."""

    rows = _IMPL._strip_csv_headers(pd.DataFrame(submission).copy())
    if "sequence_id" not in rows.columns:
        return
    text = rows["sequence_id"].astype("string").str.strip()
    invalid = text.isna() | text.eq("").fillna(False)
    if invalid.any():
        row_indices = rows.index[invalid.to_numpy(dtype=bool)].tolist()[:5]
        raise ValueError(
            "submission contains missing or blank sequence_id values: "
            f"sequence_id rows {row_indices}"
        )


def _numeric_cell_kind(value: object) -> str | None:
    """Classify scalar cell types that pandas would otherwise coerce unsafely."""

    scalar = value
    if isinstance(value, np.ma.MaskedArray):
        if bool(np.ma.getmaskarray(value).any()):
            return "masked"
        scalar = value.data
    if isinstance(scalar, np.ndarray):
        if scalar.ndim != 0:
            return "non-scalar"
        scalar = scalar.item()
    if isinstance(scalar, (bool, np.bool_)):
        return "Boolean"
    if isinstance(scalar, (complex, np.complexfloating)):
        return "complex"
    return None


def _numeric_validation_scalar(value: object) -> object:
    """Unwrap supported zero-dimensional real numeric containers."""

    scalar = value
    if isinstance(value, np.ma.MaskedArray) and not bool(np.ma.getmaskarray(value).any()):
        scalar = value.data
    if isinstance(scalar, np.ndarray) and scalar.ndim == 0:
        return scalar.item()
    return scalar


def _validate_classification_values(rows: pd.DataFrame) -> pd.Series:
    """Return canonical official class ids and report their original row indices."""

    parsed: list[int] = []
    invalid: list[str] = []
    for row_index, value in rows["Classification"].items():
        try:
            parsed.append(parse_official_classification_cell(value))
        except (TypeError, ValueError, OverflowError) as exc:
            invalid.append(f"{row_index}:{value!r} ({exc})")
    if invalid:
        preview = "; ".join(invalid[:5])
        suffix = f"; {len(invalid) - 5} more" if len(invalid) > 5 else ""
        raise ValueError(
            "submission contains invalid Classification values at rows "
            f"{preview}{suffix}"
        )
    return pd.Series(parsed, index=rows.index, dtype=int)


def _validate_numeric_rows(submission: object) -> pd.DataFrame:
    """Validate numeric rows and return safe scalar containers for projection."""

    raw_rows = pd.DataFrame(submission).copy()
    rows = _IMPL._strip_csv_headers(raw_rows.copy())
    if any(column not in rows.columns for column in _NUMERIC_COLUMNS):
        return raw_rows

    normalized = rows.copy()
    boolean_invalid: list[str] = []
    complex_invalid: list[str] = []
    masked_invalid: list[str] = []
    nonscalar_invalid: list[str] = []
    for column in _NUMERIC_COLUMNS:
        kinds = rows[column].map(_numeric_cell_kind)
        normalized[column] = rows[column].map(_numeric_validation_scalar)
        for kind, target in (
            ("Boolean", boolean_invalid),
            ("complex", complex_invalid),
            ("masked", masked_invalid),
            ("non-scalar", nonscalar_invalid),
        ):
            invalid = kinds.eq(kind).to_numpy(dtype=bool)
            if invalid.any():
                row_indices = rows.index[invalid].tolist()
                target.append(f"{column} rows {row_indices}")

    if boolean_invalid:
        details = "; ".join(boolean_invalid)
        raise ValueError(f"submission contains Boolean numeric values: {details}")
    if complex_invalid:
        details = "; ".join(complex_invalid)
        raise ValueError(f"submission contains complex numeric values: {details}")
    if masked_invalid:
        details = "; ".join(masked_invalid)
        raise ValueError(f"submission contains masked numeric values: {details}")
    if nonscalar_invalid:
        details = "; ".join(nonscalar_invalid)
        raise ValueError(f"submission contains non-scalar numeric values: {details}")

    nonfinite_invalid: list[str] = []
    for column in _NUMERIC_COLUMNS:
        numeric = pd.to_numeric(normalized[column], errors="coerce")
        finite = np.isfinite(numeric.to_numpy(dtype=float))
        if finite.all():
            continue
        row_indices = rows.index[~finite].tolist()
        nonfinite_invalid.append(f"{column} rows {row_indices}")
    if nonfinite_invalid:
        details = "; ".join(nonfinite_invalid)
        raise ValueError(f"submission contains non-finite numeric values: {details}")
    normalized["Classification"] = _validate_classification_values(normalized)
    return normalized


def _validate_unique_fixed_grid_keys(submission: object) -> None:
    """Reject duplicate normalized or official sequence/timestamp keys."""

    rows = _IMPL._strip_csv_headers(pd.DataFrame(submission).copy())
    required = {"sequence_id", "time_s"}
    if required <= set(rows.columns):
        normalized_keys = pd.DataFrame(
            {
                "sequence_id": rows["sequence_id"].astype(str),
                "time_s": pd.to_numeric(rows["time_s"], errors="coerce"),
            }
        )
    else:
        lower_to_original = {
            str(column).strip().casefold(): column for column in rows.columns
        }
        official_required = {
            "sequence",
            "timestamp",
            "position",
            "classification",
        }
        if not official_required <= set(lower_to_original):
            return
        official = normalize_official_track5_results_frame(rows)
        normalized_keys = pd.DataFrame(
            {
                "sequence_id": official["Sequence"].astype(str),
                "time_s": pd.to_numeric(official["Timestamp"], errors="coerce"),
            }
        )

    duplicate_mask = normalized_keys.duplicated(
        subset=["sequence_id", "time_s"], keep=False
    )
    if not bool(duplicate_mask.any()):
        return
    duplicate_keys = (
        normalized_keys.loc[duplicate_mask, ["sequence_id", "time_s"]]
        .drop_duplicates()
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )
    sample = ", ".join(
        f"{row.sequence_id}@{float(row.time_s):g}"
        for row in duplicate_keys.head(5).itertuples(index=False)
    )
    suffix = ", ..." if len(duplicate_keys) > 5 else ""
    raise ValueError(
        f"submission contains {len(duplicate_keys)} duplicate "
        f"(sequence_id, time_s) key(s): {sample}{suffix}"
    )


def project_track5_speed_limit(
    submission: pd.DataFrame,
    *,
    max_speed_mps: float = 60.0,
    iterations: int = 2,
    anchor_blend: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Project a trajectory after validating scalar controls and fixed-grid inputs."""

    validated_iterations = _positive_integer(iterations, name="iterations")
    validated_max_speed_mps = _reject_boolean_scalar(
        max_speed_mps,
        message="max_speed_mps must be positive and finite",
    )
    validated_anchor_blend = _reject_boolean_scalar(
        anchor_blend,
        message="anchor_blend must be finite and in [0, 1)",
    )
    _validate_sequence_ids(submission)
    validated_submission = _validate_numeric_rows(submission)
    _validate_unique_fixed_grid_keys(validated_submission)
    return _ORIGINAL_PROJECT(
        validated_submission,
        max_speed_mps=validated_max_speed_mps,
        iterations=validated_iterations,
        anchor_blend=validated_anchor_blend,
    )


_IMPL.project_track5_speed_limit = project_track5_speed_limit
_IMPL._validate_sequence_ids = _validate_sequence_ids

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_NUMERIC_COLUMNS"] = _NUMERIC_COLUMNS
globals()["_finite_scalar"] = _finite_scalar
globals()["_positive_integer"] = _positive_integer
globals()["_reject_boolean_scalar"] = _reject_boolean_scalar
globals()["_validate_sequence_ids"] = _validate_sequence_ids
globals()["_numeric_cell_kind"] = _numeric_cell_kind
globals()["_numeric_validation_scalar"] = _numeric_validation_scalar
globals()["_validate_classification_values"] = _validate_classification_values
globals()["_validate_numeric_rows"] = _validate_numeric_rows
globals()["_validate_unique_fixed_grid_keys"] = _validate_unique_fixed_grid_keys
globals()["project_track5_speed_limit"] = project_track5_speed_limit
