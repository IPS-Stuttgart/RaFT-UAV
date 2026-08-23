"""Compatibility fixes for Track 5 template resampling and diagnostics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_template_resample.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_template_resample_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load Track 5 template-resample implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RESAMPLE_ESTIMATES = _IMPL.resample_estimates_to_track5_template
_ORIGINAL_WRITE_OUTPUTS = _IMPL.write_track5_template_resample_outputs
_ORIGINAL_NORMALIZE_TEMPLATE_ROWS = _IMPL._normalize_template_rows
_ORIGINAL_RESAMPLED_POSITION = _IMPL._resampled_position
_ORIGINAL_RESAMPLED_CLASSIFICATION = _IMPL._resampled_classification

_TRUE_BOOLEAN_TEXT = frozenset({"1", "1.0", "true", "t", "yes", "y"})
_FALSE_BOOLEAN_TEXT = frozenset({"0", "0.0", "false", "f", "no", "n"})
_MISSING_BOOLEAN_TEXT = frozenset({"", "na", "nan", "none", "null", "<na>", "nat"})


def _normalize_optional_nonnegative_float(value: Any, *, field: str) -> float | None:
    """Return an optional finite non-negative scalar with a stable error."""

    if value is None:
        return None
    message = f"{field} must be a finite non-negative number"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (complex, np.complexfloating)):
        raise ValueError(message)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(message)
    return numeric


def resample_estimates_to_track5_template(
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    *,
    max_nearest_time_delta_s: float | None = None,
    resample_method="linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy="sequence-mode",
):
    """Validate time controls before resampling estimates to the template."""

    max_nearest_time_delta_s = _normalize_optional_nonnegative_float(
        max_nearest_time_delta_s,
        field="max_nearest_time_delta_s",
    )
    max_interpolation_gap_s = _normalize_optional_nonnegative_float(
        max_interpolation_gap_s,
        field="max_interpolation_gap_s",
    )
    return _ORIGINAL_RESAMPLE_ESTIMATES(
        estimates,
        template,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=classification_policy,
    )


def write_track5_template_resample_outputs(
    *,
    estimates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 0,
    max_nearest_time_delta_s: float | None = None,
    resample_method="linear",
    max_interpolation_gap_s: float | None = None,
    classification_policy="sequence-mode",
) -> dict[str, Path]:
    """Validate time controls before creating any resampling artifacts."""

    max_nearest_time_delta_s = _normalize_optional_nonnegative_float(
        max_nearest_time_delta_s,
        field="max_nearest_time_delta_s",
    )
    max_interpolation_gap_s = _normalize_optional_nonnegative_float(
        max_interpolation_gap_s,
        field="max_interpolation_gap_s",
    )
    return _ORIGINAL_WRITE_OUTPUTS(
        estimates=estimates,
        template=template,
        output_dir=output_dir,
        class_map=class_map,
        default_classification=default_classification,
        max_nearest_time_delta_s=max_nearest_time_delta_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
        classification_policy=classification_policy,
    )


def _is_complex_scalar(value: Any) -> bool:
    """Return whether a cell carries a complex scalar, including zero imaginary part."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return False
        value = value.item()
    return isinstance(value, (complex, np.complexfloating))


def _complex_scalar_mask(values: pd.Series) -> pd.Series:
    """Locate complex scalar cells without coercing them to floating point."""

    series = pd.Series(values, index=values.index)
    if pd.api.types.is_complex_dtype(series.dtype):
        return pd.Series(True, index=series.index, dtype=bool)
    return series.map(_is_complex_scalar).fillna(False).astype(bool)


def _invalid_row_message(mask: pd.Series, *, prefix: str) -> ValueError:
    invalid_indices = mask.index[mask].tolist()
    preview = ", ".join(str(index) for index in invalid_indices[:5])
    suffix = ", ..." if len(invalid_indices) > 5 else ""
    return ValueError(f"{prefix} at row indices: {preview}{suffix}")


def _alias_columns(rows: pd.DataFrame, aliases: tuple[str, ...]) -> list[Any]:
    """Return every supplied column whose case-insensitive name is a known alias."""

    normalized_aliases = {str(alias).lower() for alias in aliases}
    return [
        column
        for column in rows.columns
        if str(column).lower() in normalized_aliases
    ]


def _validate_alias_consistency(
    rows: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    normalizer: Any,
    context: str,
    field: str,
) -> None:
    """Reject rows whose redundant schema aliases normalize to different values."""

    columns = _alias_columns(rows, aliases)
    if len(columns) <= 1:
        return
    normalized = pd.concat(
        [normalizer(pd.Series(rows[column], index=rows.index)) for column in columns],
        axis=1,
    )
    conflicts = normalized.nunique(axis=1, dropna=True) > 1
    if not conflicts.any():
        return
    rendered = ", ".join(repr(str(column)) for column in columns)
    raise _invalid_row_message(
        conflicts,
        prefix=f"{context} contain conflicting {field} aliases {rendered}",
    )


def _numeric_alias_values(values: pd.Series) -> pd.Series:
    """Normalize numeric aliases without silently changing their relative meaning."""

    return pd.to_numeric(values, errors="coerce")


def _normalize_estimate_rows(estimates: pd.DataFrame) -> pd.DataFrame:
    """Normalize estimates without silently discarding malformed trajectory rows."""

    rows = pd.DataFrame(estimates).copy()
    if rows.empty:
        return pd.DataFrame(
            columns=["sequence_id", "time_s", "state_x_m", "state_y_m", "state_z_m"]
        )
    _validate_alias_consistency(
        rows,
        _IMPL.SEQUENCE_ALIASES,
        normalizer=_IMPL._normalized_sequence_values,
        context="estimates",
        field="sequence",
    )
    _validate_alias_consistency(
        rows,
        _IMPL.TIME_ALIASES,
        normalizer=_numeric_alias_values,
        context="estimates",
        field="timestamp",
    )
    _validate_alias_consistency(
        rows,
        _IMPL.CLASSIFICATION_ALIASES,
        normalizer=_IMPL._normalized_classification_values,
        context="estimates",
        field="classification",
    )
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    time_column = _IMPL._first_present(rows, _IMPL.TIME_ALIASES)
    coord_columns = _IMPL._coordinate_columns(rows)
    classification_column = _IMPL._first_present(rows, _IMPL.CLASSIFICATION_ALIASES)
    if sequence_column is None or time_column is None:
        raise ValueError("estimates must contain sequence and time columns")

    normalized_sequence = _IMPL._normalized_sequence_values(rows[sequence_column])
    valid_sequence = normalized_sequence.notna()
    complex_numeric = pd.Series(False, index=rows.index, dtype=bool)
    for column in (*_alias_columns(rows, _IMPL.TIME_ALIASES), *coord_columns):
        complex_numeric |= _complex_scalar_mask(rows[column])
    if (valid_sequence & complex_numeric).any():
        raise _invalid_row_message(
            valid_sequence & complex_numeric,
            prefix="estimates contain complex time or position values",
        )

    out = pd.DataFrame(
        {
            "sequence_id": normalized_sequence,
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "state_x_m": pd.to_numeric(rows[coord_columns[0]], errors="coerce"),
            "state_y_m": pd.to_numeric(rows[coord_columns[1]], errors="coerce"),
            "state_z_m": pd.to_numeric(rows[coord_columns[2]], errors="coerce"),
        }
    )
    if classification_column is not None:
        out["classification"] = _IMPL._normalized_classification_values(
            rows[classification_column]
        )
    finite_numeric = np.isfinite(
        out[["time_s", "state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    ).all(axis=1)
    invalid_numeric = valid_sequence & ~finite_numeric
    if invalid_numeric.any():
        raise _invalid_row_message(
            invalid_numeric,
            prefix="estimates contain non-finite or non-numeric time or position values",
        )
    return (
        out.loc[valid_sequence & finite_numeric]
        .sort_values(["sequence_id", "time_s"], kind="mergesort")
        .reset_index(drop=True)
    )


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    """Reject ambiguous or complex programmatic template timestamps."""

    rows = pd.DataFrame(template).copy()
    if rows.empty:
        return _ORIGINAL_NORMALIZE_TEMPLATE_ROWS(rows)
    _validate_alias_consistency(
        rows,
        _IMPL.SEQUENCE_ALIASES,
        normalizer=_IMPL._normalized_sequence_values,
        context="template",
        field="sequence",
    )
    _validate_alias_consistency(
        rows,
        _IMPL.TIME_ALIASES,
        normalizer=_numeric_alias_values,
        context="template",
        field="timestamp",
    )
    sequence_column = _IMPL._first_present(rows, _IMPL.SEQUENCE_ALIASES)
    time_column = _IMPL._first_present(rows, _IMPL.TIME_ALIASES)
    if sequence_column is None or time_column is None:
        return _ORIGINAL_NORMALIZE_TEMPLATE_ROWS(rows)

    valid_sequence = _IMPL._normalized_sequence_values(rows[sequence_column]).notna()
    complex_time = pd.Series(False, index=rows.index, dtype=bool)
    for column in _alias_columns(rows, _IMPL.TIME_ALIASES):
        complex_time |= _complex_scalar_mask(rows[column])
    if (valid_sequence & complex_time).any():
        raise _invalid_row_message(
            valid_sequence & complex_time,
            prefix="template contains complex timestamp values",
        )
    return _ORIGINAL_NORMALIZE_TEMPLATE_ROWS(rows)


def _unique_time_rows(group: pd.DataFrame) -> pd.DataFrame:
    """Apply one deterministic keep-last rule to every timestamp-dependent field."""

    rows = pd.DataFrame(group).copy()
    if rows.empty or "time_s" not in rows.columns:
        return rows
    return (
        rows.sort_values("time_s", kind="mergesort")
        .drop_duplicates("time_s", keep="last")
        .reset_index(drop=True)
    )


def _boolean_diagnostic_error(
    *,
    column: str,
    row_index: object,
    value: object,
) -> ValueError:
    """Build one stable error for malformed Boolean diagnostic cells."""

    return ValueError(
        f"{column} must contain Boolean diagnostics encoded as true/false or 1/0; "
        f"got {value!r} at row index {row_index!r}"
    )


def _unwrap_boolean_scalar(
    value: object,
    *,
    column: str,
    row_index: object,
) -> object:
    """Unwrap scalar NumPy containers without exposing masks or cycling forever."""

    seen: set[int] = set()
    while isinstance(value, np.ndarray):
        if np.ma.is_masked(value):
            return pd.NA
        if value.ndim != 0:
            raise _boolean_diagnostic_error(
                column=column,
                row_index=row_index,
                value=value,
            )
        marker = id(value)
        if marker in seen:
            raise _boolean_diagnostic_error(
                column=column,
                row_index=row_index,
                value="cyclic scalar container",
            )
        seen.add(marker)
        value = value.item()
    return value


def _boolean_diagnostic_value(
    value: object,
    *,
    column: str,
    row_index: object,
) -> bool:
    """Parse one strict Boolean diagnostic cell."""

    value = _unwrap_boolean_scalar(
        value,
        column=column,
        row_index=row_index,
    )
    if np.ma.is_masked(value) or value is None or value is pd.NA or value is pd.NaT:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (complex, np.complexfloating)):
        raise _boolean_diagnostic_error(
            column=column,
            row_index=row_index,
            value=value,
        )
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_BOOLEAN_TEXT:
            return True
        if normalized in _FALSE_BOOLEAN_TEXT or normalized in _MISSING_BOOLEAN_TEXT:
            return False
        raise _boolean_diagnostic_error(
            column=column,
            row_index=row_index,
            value=value,
        )

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _boolean_diagnostic_error(
            column=column,
            row_index=row_index,
            value=value,
        ) from exc
    if not np.isfinite(numeric) or numeric not in (0.0, 1.0):
        raise _boolean_diagnostic_error(
            column=column,
            row_index=row_index,
            value=value,
        )
    return bool(numeric)


def _bool_column(rows: pd.DataFrame, column: str) -> pd.Series:
    """Normalize strict Boolean diagnostics, including CSV-style ``1.0`` / ``0.0``."""

    if column not in rows.columns:
        return pd.Series(False, index=rows.index, dtype=bool)
    values = pd.Series(rows[column], index=rows.index)
    normalized = [
        _boolean_diagnostic_value(
            value,
            column=column,
            row_index=row_index,
        )
        for row_index, value in values.items()
    ]
    return pd.Series(normalized, index=rows.index, dtype=bool)


def _resampled_position(
    group: pd.DataFrame,
    time_s: float,
    *,
    resample_method,
    max_interpolation_gap_s,
):
    return _ORIGINAL_RESAMPLED_POSITION(
        _unique_time_rows(group),
        time_s,
        resample_method=resample_method,
        max_interpolation_gap_s=max_interpolation_gap_s,
    )


def _resampled_classification(
    group: pd.DataFrame,
    time_s: float,
    *,
    classification_policy,
):
    return _ORIGINAL_RESAMPLED_CLASSIFICATION(
        _unique_time_rows(group),
        time_s,
        classification_policy=classification_policy,
    )


_IMPL.resample_estimates_to_track5_template = resample_estimates_to_track5_template
_IMPL.write_track5_template_resample_outputs = write_track5_template_resample_outputs
_IMPL._normalize_estimate_rows = _normalize_estimate_rows
_IMPL._normalize_template_rows = _normalize_template_rows
_IMPL._bool_column = _bool_column
_IMPL._resampled_position = _resampled_position
_IMPL._resampled_classification = _resampled_classification

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_optional_nonnegative_float"] = _normalize_optional_nonnegative_float
globals()["resample_estimates_to_track5_template"] = resample_estimates_to_track5_template
globals()["write_track5_template_resample_outputs"] = write_track5_template_resample_outputs
globals()["_is_complex_scalar"] = _is_complex_scalar
globals()["_complex_scalar_mask"] = _complex_scalar_mask
globals()["_invalid_row_message"] = _invalid_row_message
globals()["_alias_columns"] = _alias_columns
globals()["_validate_alias_consistency"] = _validate_alias_consistency
globals()["_numeric_alias_values"] = _numeric_alias_values
globals()["_normalize_estimate_rows"] = _normalize_estimate_rows
globals()["_normalize_template_rows"] = _normalize_template_rows
globals()["_boolean_diagnostic_error"] = _boolean_diagnostic_error
globals()["_unwrap_boolean_scalar"] = _unwrap_boolean_scalar
globals()["_boolean_diagnostic_value"] = _boolean_diagnostic_value
globals()["_bool_column"] = _bool_column
globals()["_resampled_position"] = _resampled_position
globals()["_resampled_classification"] = _resampled_classification

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
