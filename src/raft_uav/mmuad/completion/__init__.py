"""Compatibility validation for MMUAD trajectory completion.

The maintained implementation lives in the sibling ``completion.py`` module.
This package preserves the public import path while rejecting malformed scalar
controls and normalizing missing-like sequence identifiers consistently before
completion or summary generation starts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "completion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._completion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load completion implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_COMPLETION_SUMMARY = _IMPL.completion_summary
_ORIGINAL_COMPLETION_RESULT_ROWS = _IMPL._completion_result_rows
_ORIGINAL_COMPLETION_TEMPLATE_ROWS = _IMPL._completion_template_rows
_MISSING_SEQUENCE_ID_STRINGS = {"", "nan", "none", "<na>", "nat"}
_SEQUENCE_ID_ALIASES = (
    "sequence_id",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "id",
    "name",
)


def _normalize_max_interpolation_gap_s(value: object) -> float:
    """Return a finite non-negative real scalar interpolation gap."""

    message = "max_interpolation_gap_s must be a finite non-negative number"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or array.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        gap = float(array.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError(message)
    return gap


def _completion_sequence_values(value: object) -> pd.Series | None:
    """Return the raw sequence-ID column used by completion, when available."""

    if isinstance(value, pd.DataFrame):
        frame = value
    else:
        frame = getattr(value, "rows", None)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None

    columns: dict[str, object] = {}
    for column in frame.columns:
        columns.setdefault(str(column).strip().casefold(), column)
    for alias in _SEQUENCE_ID_ALIASES:
        column = columns.get(alias)
        if column is None:
            continue
        values = frame.loc[:, column]
        if isinstance(values, pd.DataFrame):
            return None
        return values
    return None


def _validate_completion_sequence_metadata(value: object, *, name: str) -> None:
    """Reject ambiguous mixes of explicit and missing completion sequence IDs."""

    values = _completion_sequence_values(value)
    if values is None or values.empty:
        return
    missing = values.isna()
    text = values.where(~missing, "").astype(str).str.strip().str.casefold()
    missing = missing | text.isin(_MISSING_SEQUENCE_ID_STRINGS)
    if bool(missing.any()) and bool((~missing).any()):
        explicit_ids = set(text.loc[~missing].tolist())
        if explicit_ids == {"default"}:
            return
        raise ValueError(
            f"{name} sequence IDs are partially missing; provide sequence IDs "
            "for every row or omit them for the entire table"
        )


def _completion_result_rows(results: object) -> pd.DataFrame:
    """Validate result sequence metadata before legacy normalization."""

    _validate_completion_sequence_metadata(results, name="completion results")
    return _ORIGINAL_COMPLETION_RESULT_ROWS(results)


def _completion_template_rows(truth_or_template: object) -> pd.DataFrame:
    """Canonicalize missing-like template sequence ids before grouping."""

    _validate_completion_sequence_metadata(
        truth_or_template,
        name="completion template",
    )
    rows = _ORIGINAL_COMPLETION_TEMPLATE_ROWS(truth_or_template)
    if rows.empty or "sequence_id" not in rows.columns:
        return rows

    normalized = rows.copy()
    text = normalized["sequence_id"].where(
        normalized["sequence_id"].notna(),
        "default",
    )
    text = text.astype(str).str.strip()
    missing = text.str.casefold().isin(_MISSING_SEQUENCE_ID_STRINGS)
    normalized["sequence_id"] = text.where(~missing, "default")
    return (
        normalized.drop_duplicates()
        .sort_values(["sequence_id", "time_s"])
        .reset_index(drop=True)
    )


def _normalize_requested_count(value: object, *, completed_count: int) -> int:
    """Return an exact count consistent with the completed result rows."""

    requested = optional_int(value)
    if requested is None or requested < 0:
        raise ValueError("requested_count must be a non-negative integer")
    if requested < completed_count:
        raise ValueError(
            f"requested_count ({requested}) cannot be smaller than "
            f"completed_count ({completed_count})"
        )
    return requested


def completion_summary(result, *, requested_count: object | None = None):
    """Summarize completion after validating an explicit requested count."""

    normalized_requested_count = None
    if requested_count is not None:
        normalized_requested_count = _normalize_requested_count(
            requested_count,
            completed_count=len(result.rows),
        )
    return _ORIGINAL_COMPLETION_SUMMARY(
        result,
        requested_count=normalized_requested_count,
    )


_IMPL._normalize_max_interpolation_gap_s = _normalize_max_interpolation_gap_s
_IMPL._completion_result_rows = _completion_result_rows
_IMPL._completion_template_rows = _completion_template_rows
_IMPL.completion_summary = completion_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_max_interpolation_gap_s"] = _normalize_max_interpolation_gap_s
globals()["_completion_result_rows"] = _completion_result_rows
globals()["_completion_template_rows"] = _completion_template_rows
globals()["_normalize_requested_count"] = _normalize_requested_count
globals()["completion_summary"] = completion_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]