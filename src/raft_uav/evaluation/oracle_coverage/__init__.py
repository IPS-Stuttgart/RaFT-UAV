"""Compatibility fixes for oracle-coverage identifiers and numeric controls.

The maintained implementation lives in the sibling ``oracle_coverage.py``
module. This package preserves the public import path while preventing
fractional identifiers from being truncated, large exact identifiers from being
rounded through binary floating point, and malformed radar standard deviations
from silently changing candidate scoring.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.numeric import optional_float as _shared_optional_float
from raft_uav.numeric import optional_int as _shared_optional_int

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

_ORIGINAL_BUILD_ORACLE_CANDIDATE_COVERAGE = _IMPL.build_oracle_candidate_coverage
_IDENTIFIER_KEY_COLUMNS = frozenset({"frame_index", "track_index", "track_id"})
_CANDIDATE_KEY_COLUMNS = (
    "frame_index",
    "track_index",
    "track_id",
    "time_s",
    "east_m",
    "north_m",
    "up_m",
)


def _positive_standard_deviation(name: str, value: object) -> float:
    """Return a finite positive real scalar standard deviation."""

    standard_deviation = _shared_optional_float(value)
    if standard_deviation is None or standard_deviation <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return standard_deviation


def build_oracle_candidate_coverage(
    *,
    radar: Any,
    truth: Any,
    rf_measurements: Any = (),
    candidate_catprob_threshold: float | None = 0.5,
    config: Any = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: object = 25.0,
    radar_z_std_m: object = 35.0,
    truth_time_gate_s: float | None = 1.0,
    gate_probabilities_by_source: Any = None,
    gate_thresholds_by_source: Any = None,
    safety_gate_probabilities_by_source: Any = None,
    safety_gate_thresholds_by_source: Any = None,
    robust_update_by_source: Any = None,
    inflation_alpha_by_source: Any = None,
    max_residual_norms_by_source: Any = None,
) -> Any:
    """Build oracle coverage after validating radar measurement uncertainty."""

    validated_xy_std_m = _positive_standard_deviation(
        "radar_xy_std_m",
        radar_xy_std_m,
    )
    validated_z_std_m = _positive_standard_deviation(
        "radar_z_std_m",
        radar_z_std_m,
    )
    return _ORIGINAL_BUILD_ORACLE_CANDIDATE_COVERAGE(
        radar=radar,
        truth=truth,
        rf_measurements=rf_measurements,
        candidate_catprob_threshold=candidate_catprob_threshold,
        config=config,
        acceleration_std_mps2=acceleration_std_mps2,
        radar_xy_std_m=validated_xy_std_m,
        radar_z_std_m=validated_z_std_m,
        truth_time_gate_s=truth_time_gate_s,
        gate_probabilities_by_source=gate_probabilities_by_source,
        gate_thresholds_by_source=gate_thresholds_by_source,
        safety_gate_probabilities_by_source=safety_gate_probabilities_by_source,
        safety_gate_thresholds_by_source=safety_gate_thresholds_by_source,
        robust_update_by_source=robust_update_by_source,
        inflation_alpha_by_source=inflation_alpha_by_source,
        max_residual_norms_by_source=max_residual_norms_by_source,
    )


def _optional_int(value: object) -> int | None:
    """Return an exact integer-equivalent scalar without truncation or rounding."""

    return _shared_optional_int(value)


def _candidate_key(row: Any) -> tuple[tuple[str, object], ...]:
    """Build a candidate key while preserving exact identifier identity."""

    columns = [column for column in _CANDIDATE_KEY_COLUMNS if column in row.index]
    key: list[tuple[str, object]] = []
    for column in columns:
        value = row[column]
        if column in _IDENTIFIER_KEY_COLUMNS:
            exact = _optional_int(value)
            stable = exact if exact is not None else str(value)
        else:
            stable = _IMPL._stable_value(value)
        key.append((column, stable))
    return tuple(key)


def _event_key(candidates: Any, time_s: float) -> str:
    """Report the first valid frame identifier without a float round-trip."""

    if "frame_index" in candidates.columns and not candidates.empty:
        for value in candidates["frame_index"].tolist():
            frame_index = _optional_int(value)
            if frame_index is not None:
                return f"frame_index:{frame_index}"
    return f"time_s:{float(time_s):.9f}"


_IMPL.build_oracle_candidate_coverage = build_oracle_candidate_coverage
_IMPL._optional_int = _optional_int
_IMPL._candidate_key = _candidate_key
_IMPL._event_key = _event_key

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_positive_standard_deviation"] = _positive_standard_deviation
globals()["build_oracle_candidate_coverage"] = build_oracle_candidate_coverage
globals()["_optional_int"] = _optional_int
globals()["_candidate_key"] = _candidate_key
globals()["_event_key"] = _event_key

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
