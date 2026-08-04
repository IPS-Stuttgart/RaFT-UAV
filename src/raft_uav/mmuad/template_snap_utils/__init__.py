"""Compatibility fixes for strict Track 5 template-snap inputs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "template_snap_utils.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._template_snap_utils_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load template-snap utilities from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_LOAD_OFFICIAL_TRACK5_RESULTS_FRAME_FROM_FRAME = (
    _IMPL.load_official_track5_results_frame_from_frame
)
_ORIGINAL_NORMALIZE_TEMPLATE_ROWS = _IMPL._normalize_template_rows


def _require_unique_normalized_columns(
    frame: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Reject physical headers that collapse to the same lookup key."""

    columns_by_key: dict[str, list[object]] = {}
    for column in frame.columns:
        key = str(column).strip().lower()
        columns_by_key.setdefault(key, []).append(column)

    collisions = [
        columns
        for columns in columns_by_key.values()
        if len(columns) > 1
    ]
    if not collisions:
        return

    rendered = "; ".join(
        ", ".join(repr(str(column)) for column in columns)
        for columns in collisions
    )
    raise ValueError(
        f"{context} contains ambiguous columns after trimming whitespace "
        f"and case-folding: {rendered}"
    )


def _is_boolean_scalar_value(value: object) -> bool:
    """Return whether a cell is a Python, NumPy, or zero-dimensional Boolean."""

    if isinstance(value, (bool, np.bool_)):
        return True
    return bool(
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and np.issubdtype(value.dtype, np.bool_)
    )


def _is_complex_scalar_value(value: object) -> bool:
    """Return whether a cell is a Python, NumPy, or zero-dimensional complex scalar."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    return bool(
        isinstance(value, np.ndarray)
        and value.ndim == 0
        and np.issubdtype(value.dtype, np.complexfloating)
    )


def _validate_real_timestamp_cells(
    values: pd.Series,
    *,
    context: str,
) -> None:
    """Reject timestamp cells that pandas would reinterpret as real numbers."""

    raw = pd.Series(values)
    boolean_mask = raw.map(_is_boolean_scalar_value)
    if boolean_mask.any():
        row_position = int(np.flatnonzero(boolean_mask.to_numpy())[0])
        bad_value = raw.iloc[row_position]
        raise ValueError(
            f"{context} Timestamp values must be real numbers, not booleans; "
            f"got {bad_value!r} at row position {row_position}"
        )

    complex_mask = raw.map(_is_complex_scalar_value)
    if complex_mask.any():
        row_position = int(np.flatnonzero(complex_mask.to_numpy())[0])
        bad_value = raw.iloc[row_position]
        raise ValueError(
            f"{context} Timestamp values must be real numbers, not complex numbers; "
            f"got {bad_value!r} at row position {row_position}"
        )


def _timestamp_column(
    frame: pd.DataFrame,
    *,
    aliases: tuple[str, ...],
) -> object | None:
    """Return the first normalized timestamp alias without truthiness coercion."""

    lower = {str(column).strip().lower(): column for column in frame.columns}
    for alias in aliases:
        column = lower.get(alias)
        if column is not None:
            return column
    return None


def load_official_track5_results_frame_from_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize official results only when headers and timestamp scalars are safe."""

    normalized_frame = pd.DataFrame(frame)
    _require_unique_normalized_columns(
        normalized_frame,
        context="official Track 5 results",
    )
    timestamp_column = _timestamp_column(normalized_frame, aliases=("timestamp",))
    if timestamp_column is not None:
        _validate_real_timestamp_cells(
            normalized_frame[timestamp_column],
            context="official Track 5 results",
        )
    return _ORIGINAL_LOAD_OFFICIAL_TRACK5_RESULTS_FRAME_FROM_FRAME(frame)


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    """Normalize template rows only when headers and timestamp scalars are safe."""

    normalized_template = pd.DataFrame(template)
    _require_unique_normalized_columns(
        normalized_template,
        context="Track 5 template",
    )
    timestamp_column = _timestamp_column(
        normalized_template,
        aliases=("timestamp", "time_s"),
    )
    if timestamp_column is not None:
        _validate_real_timestamp_cells(
            normalized_template[timestamp_column],
            context="Track 5 template",
        )
    return _ORIGINAL_NORMALIZE_TEMPLATE_ROWS(template)


def _is_complex_classification_value(value: object) -> bool:
    """Return whether a classification cell has a complex scalar dtype."""

    return _is_complex_scalar_value(value)


def _integer_classification_values(values: pd.Series) -> pd.Series:
    """Return exact finite integer-valued official classification cells."""

    raw = pd.Series(values)
    boolean_mask = raw.map(_is_boolean_scalar_value)
    if boolean_mask.any():
        row_index = int(np.flatnonzero(boolean_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids, not booleans; "
            f"got {bad_value!r}"
        )

    complex_mask = raw.map(_is_complex_classification_value)
    if complex_mask.any():
        row_index = int(np.flatnonzero(complex_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids, not complex numbers; "
            f"got {bad_value!r}"
        )

    numbers = pd.to_numeric(raw, errors="coerce")
    bad_text_mask = numbers.isna() & raw.notna()
    if bad_text_mask.any():
        row_index = int(np.flatnonzero(bad_text_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids; "
            f"got {bad_value!r}"
        )

    numeric = numbers.to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    nonfinite = numbers.notna().to_numpy() & ~finite
    if nonfinite.any():
        row_index = int(np.flatnonzero(nonfinite)[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be finite integer ids; "
            f"got {bad_value!r}"
        )

    integer_like = finite & (numeric == np.rint(numeric))
    fractional = finite & ~integer_like
    if fractional.any():
        row_index = int(np.flatnonzero(fractional)[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids; "
            f"got {bad_value!r}"
        )

    invalid_domain = np.zeros_like(finite, dtype=bool)
    if integer_like.any():
        integer_values = np.rint(numeric[integer_like]).astype(int)
        invalid_domain[integer_like] = ~np.isin(
            integer_values,
            list(_IMPL.OFFICIAL_TRACK5_CLASS_IDS),
        )
    if invalid_domain.any():
        row_index = int(np.flatnonzero(invalid_domain)[0])
        class_id = int(np.rint(numeric[row_index]))
        allowed = ", ".join(
            str(item) for item in sorted(_IMPL.OFFICIAL_TRACK5_CLASS_IDS)
        )
        raise ValueError(
            "official MMUAD Classification values must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return numbers


_IMPL.load_official_track5_results_frame_from_frame = (
    load_official_track5_results_frame_from_frame
)
_IMPL._normalize_template_rows = _normalize_template_rows
_IMPL._integer_classification_values = _integer_classification_values

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_require_unique_normalized_columns"] = _require_unique_normalized_columns
globals()["_is_boolean_scalar_value"] = _is_boolean_scalar_value
globals()["_is_complex_scalar_value"] = _is_complex_scalar_value
globals()["_validate_real_timestamp_cells"] = _validate_real_timestamp_cells
globals()["_timestamp_column"] = _timestamp_column
globals()["load_official_track5_results_frame_from_frame"] = (
    load_official_track5_results_frame_from_frame
)
globals()["_normalize_template_rows"] = _normalize_template_rows
globals()["_is_complex_classification_value"] = _is_complex_classification_value
globals()["_integer_classification_values"] = _integer_classification_values

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
