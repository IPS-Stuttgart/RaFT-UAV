"""Compatibility fix for oracle-coverage candidate identifier normalization.

The maintained implementation lives in the sibling ``oracle_coverage.py``
module. This package preserves the public import path while preventing exact
integer candidate identifiers from being truncated or rounded in diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.numeric import optional_int as _exact_optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "oracle_coverage.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._oracle_coverage_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load oracle coverage implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ORACLE_COVERAGE_ROW = _IMPL._oracle_coverage_row
_IDENTIFIER_COLUMNS = ("frame_index", "track_index", "track_id")


def _optional_int(value: object) -> int | None:
    """Return an exact integer-equivalent scalar without float round-trips."""

    return _exact_optional_int(value)


def _stable_value(value: object) -> object:
    """Return a deterministic candidate-key value without integer rounding."""

    integer = _optional_int(value)
    if integer is not None:
        return integer
    number = _IMPL._optional_float(value)
    if number is None:
        return str(value)
    return round(float(number), 9)


def _event_key(candidates: pd.DataFrame, time_s: float) -> str:
    """Build an event label without truncating or rounding valid frame IDs."""

    if "frame_index" in candidates.columns and not candidates.empty:
        for value in candidates["frame_index"].tolist():
            integer = _optional_int(value)
            if integer is not None:
                return f"frame_index:{integer}"
            number = _IMPL._optional_float(value)
            if number is not None:
                rounded = round(float(number), 9)
                normalized = int(rounded) if rounded.is_integer() else rounded
                return f"frame_index:{normalized}"
    return f"time_s:{float(time_s):.9f}"


def _identifier_safe_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep identifier scalars exact when pandas materializes mixed-type rows."""

    normalized = candidates.copy()
    for column in _IDENTIFIER_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.Series(
                normalized[column].tolist(),
                index=normalized.index,
                dtype=object,
            )
    return normalized


def _oracle_coverage_row(
    *,
    event_index: int,
    event: Mapping[str, object],
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    anchor: object | None,
    covariance: Any,
    candidate_catprob_threshold: float | None,
    config: Any,
    truth_time_gate_s: float | None,
    previous_miss_streak: int,
) -> tuple[dict[str, Any], bool]:
    """Build one coverage row without pandas coercing identifiers to floats."""

    return _ORIGINAL_ORACLE_COVERAGE_ROW(
        event_index=event_index,
        event=event,
        candidates=_identifier_safe_candidates(candidates),
        truth=truth,
        anchor=anchor,
        covariance=covariance,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
        truth_time_gate_s=truth_time_gate_s,
        previous_miss_streak=previous_miss_streak,
    )


_IMPL._optional_int = _optional_int
_IMPL._stable_value = _stable_value
_IMPL._event_key = _event_key
_IMPL._oracle_coverage_row = _oracle_coverage_row

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_optional_int"] = _optional_int
globals()["_stable_value"] = _stable_value
globals()["_event_key"] = _event_key
globals()["_identifier_safe_candidates"] = _identifier_safe_candidates
globals()["_oracle_coverage_row"] = _oracle_coverage_row

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
