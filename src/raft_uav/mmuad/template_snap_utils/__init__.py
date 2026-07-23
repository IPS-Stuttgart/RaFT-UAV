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


def load_official_track5_results_frame_from_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize official results only when every header is unambiguous."""

    _require_unique_normalized_columns(
        pd.DataFrame(frame),
        context="official Track 5 results",
    )
    return _ORIGINAL_LOAD_OFFICIAL_TRACK5_RESULTS_FRAME_FROM_FRAME(frame)


def _normalize_template_rows(template: pd.DataFrame) -> pd.DataFrame:
    """Normalize template rows only when every header is unambiguous."""

    _require_unique_normalized_columns(
        pd.DataFrame(template),
        context="Track 5 template",
    )
    return _ORIGINAL_NORMALIZE_TEMPLATE_ROWS(template)


def _integer_classification_values(values: pd.Series) -> pd.Series:
    """Return exact finite integer-valued official classification cells."""

    raw = pd.Series(values)
    boolean_mask = raw.map(lambda value: isinstance(value, (bool, np.bool_)))
    if boolean_mask.any():
        row_index = int(np.flatnonzero(boolean_mask.to_numpy())[0])
        bad_value = raw.iloc[row_index]
        raise ValueError(
            "official MMUAD Classification values must be integer ids, not booleans; "
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
globals()["load_official_track5_results_frame_from_frame"] = (
    load_official_track5_results_frame_from_frame
)
globals()["_normalize_template_rows"] = _normalize_template_rows
globals()["_integer_classification_values"] = _integer_classification_values

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
