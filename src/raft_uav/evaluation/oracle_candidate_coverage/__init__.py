"""Compatibility validation for oracle candidate-coverage inputs.

The maintained implementation lives in the sibling
``oracle_candidate_coverage.py`` module. This package preserves the public
import path while rejecting malformed truth-matching gates, radar standard
deviations, and explicit tracklet configurations and preventing fractional
candidate identifiers from being silently truncated.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.numeric import optional_float
from raft_uav.numeric import optional_int as _shared_optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "oracle_candidate_coverage.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._oracle_candidate_coverage_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load oracle candidate coverage from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_DIAGNOSTICS = _IMPL.build_oracle_candidate_coverage_diagnostics


def _nonnegative_finite_scalar(value: object, *, name: str) -> float:
    """Return a finite non-negative scalar or raise a field-specific error."""

    normalized = optional_float(value)
    if normalized is None or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return normalized


def _positive_finite_scalar(value: object, *, name: str) -> float:
    """Return a finite positive scalar or raise a field-specific error."""

    normalized = optional_float(value)
    if normalized is None or normalized <= 0.0:
        raise ValueError(f"{name} must be a finite positive real scalar")
    return normalized


def _validated_tracklet_config(config: object | None) -> Any:
    """Return a valid explicit tracklet configuration or the default instance."""

    config_type = _IMPL._base_tracklet.TrackletViterbiAssociationConfig
    if config is None:
        return config_type()
    if not isinstance(config, config_type):
        raise ValueError(
            "config must be a TrackletViterbiAssociationConfig instance or None"
        )
    return config


def _optional_int(value: object) -> int | None:
    """Return an exact integer-equivalent scalar without truncation."""

    return _shared_optional_int(value)


def build_oracle_candidate_coverage_diagnostics(
    *,
    radar: pd.DataFrame,
    truth: pd.DataFrame,
    rf_measurements: list[Any] | None = None,
    acceleration_std_mps2: float = 4.0,
    radar_xy_std_m: float = 25.0,
    radar_z_std_m: float = 35.0,
    candidate_catprob_threshold: float | None = 0.5,
    truth_time_gate_s: float = 1.0,
    truth_gate_m: float | None = None,
    config: object | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build coverage diagnostics after validating public controls."""

    normalized_config = _validated_tracklet_config(config)
    normalized_time_gate = _nonnegative_finite_scalar(
        truth_time_gate_s,
        name="truth_time_gate_s",
    )
    normalized_distance_gate = (
        None
        if truth_gate_m is None
        else _nonnegative_finite_scalar(truth_gate_m, name="truth_gate_m")
    )
    normalized_xy_std = _positive_finite_scalar(
        radar_xy_std_m,
        name="radar_xy_std_m",
    )
    normalized_z_std = _positive_finite_scalar(
        radar_z_std_m,
        name="radar_z_std_m",
    )
    return _ORIGINAL_BUILD_DIAGNOSTICS(
        radar=radar,
        truth=truth,
        rf_measurements=rf_measurements,
        acceleration_std_mps2=acceleration_std_mps2,
        radar_xy_std_m=normalized_xy_std,
        radar_z_std_m=normalized_z_std,
        candidate_catprob_threshold=candidate_catprob_threshold,
        truth_time_gate_s=normalized_time_gate,
        truth_gate_m=normalized_distance_gate,
        config=normalized_config,
    )


_IMPL._optional_int = _optional_int
_IMPL.build_oracle_candidate_coverage_diagnostics = (
    build_oracle_candidate_coverage_diagnostics
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_nonnegative_finite_scalar"] = _nonnegative_finite_scalar
globals()["_positive_finite_scalar"] = _positive_finite_scalar
globals()["_validated_tracklet_config"] = _validated_tracklet_config
globals()["_optional_int"] = _optional_int
globals()["build_oracle_candidate_coverage_diagnostics"] = (
    build_oracle_candidate_coverage_diagnostics
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
