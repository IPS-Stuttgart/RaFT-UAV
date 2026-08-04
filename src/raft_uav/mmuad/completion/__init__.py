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
_ORIGINAL_COMPLETION_TEMPLATE_ROWS = _IMPL._completion_template_rows
_MISSING_SEQUENCE_ID_STRINGS = {"", "nan", "none", "<na>", "nat"}


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


def _is_boolean_template_time(value: object) -> bool:
    """Return whether a scalar-like template time contains a Boolean."""

    scalar = value
    seen_array_ids: set[int] = set()
    while True:
        if isinstance(scalar, (bool, np.bool_)):
            return True
        if isinstance(scalar, np.ndarray):
            if scalar.ndim != 0:
                return False
            identity = id(scalar)
            if identity in seen_array_ids:
                return False
            seen_array_ids.add(identity)
            if np.issubdtype(scalar.dtype, np.bool_):
                return True
            try:
                scalar = scalar.item()
            except ValueError:
                return False
            continue
        if isinstance(scalar, np.generic):
            scalar = scalar.item()
            continue
        return False


def _reject_boolean_template_timestamps(truth_or_template: object) -> None:
    """Reject template times that pandas would silently reinterpret as 0 or 1."""

    raw = (
        truth_or_template.rows
        if isinstance(truth_or_template, _IMPL.TruthFrame)
        else pd.DataFrame(truth_or_template)
    )
    normalized = _IMPL.normalize_time_column_aliases(
        pd.DataFrame(raw).copy(),
        target="time_s",
    )
    if "time_s" not in normalized.columns:
        return
    boolean_mask = normalized["time_s"].map(_is_boolean_template_time).to_numpy(
        dtype=bool
    )
    if not bool(boolean_mask.any()):
        return
    row_position = int(np.flatnonzero(boolean_mask)[0])
    raise ValueError(
        "completion template time_s must not contain Boolean values; "
        f"row position {row_position}"
    )


def _completion_template_rows(truth_or_template: object) -> pd.DataFrame:
    """Canonicalize missing-like ids after validating template time scalars."""

    _reject_boolean_template_timestamps(truth_or_template)
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
globals()["_is_boolean_template_time"] = _is_boolean_template_time
globals()["_reject_boolean_template_timestamps"] = (
    _reject_boolean_template_timestamps
)
globals()["_completion_template_rows"] = _completion_template_rows
globals()["_normalize_requested_count"] = _normalize_requested_count
globals()["completion_summary"] = completion_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
