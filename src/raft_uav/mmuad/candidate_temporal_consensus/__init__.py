"""Strict validation for MMUAD temporal-consensus configuration.

The maintained implementation lives in the sibling
``candidate_temporal_consensus.py`` module. This package preserves the public
import path while rejecting malformed settings before they can silently select
defaults, disable temporal gating, or propagate non-finite consensus scores.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_temporal_consensus.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_temporal_consensus_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load temporal-consensus implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_TEMPORAL_CANDIDATE_CONSENSUS = _IMPL.add_temporal_candidate_consensus
_POSITIVE_CONFIG_FIELDS = (
    "max_time_gap_s",
    "max_speed_mps",
    "distance_scale_m",
    "acceleration_scale_mps2",
)
_FINITE_CONFIG_FIELDS = (
    *_POSITIVE_CONFIG_FIELDS,
    "base_score_weight",
    "backward_support_weight",
    "forward_support_weight",
    "bidirectional_bonus",
    "interpolation_weight",
    "acceleration_weight",
    "source_diversity_bonus",
    "branch_diversity_bonus",
)


def _unwrapped_real_scalar(value: Any, *, field_name: str) -> Any:
    """Return a scalar payload without lossy nested-container coercion."""

    message = f"{field_name} must be a finite real scalar"
    scalar = value
    seen_arrays: set[int] = set()
    while isinstance(scalar, (np.ndarray, np.generic)):
        if np.ma.is_masked(scalar):
            raise ValueError(message)
        if isinstance(scalar, np.ndarray):
            if scalar.ndim != 0:
                raise ValueError(message)
            marker = id(scalar)
            if marker in seen_arrays:
                raise ValueError(message)
            seen_arrays.add(marker)
        scalar = scalar.item()

    if (
        np.ma.is_masked(scalar)
        or isinstance(scalar, (bool, np.bool_))
        or isinstance(scalar, (complex, np.complexfloating))
    ):
        raise ValueError(message)
    try:
        scalar_array = np.asarray(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar_array.ndim != 0 or np.iscomplexobj(scalar_array):
        raise ValueError(message)
    return scalar


def _finite_real_scalar(value: Any, *, field_name: str) -> float:
    """Return one finite real scalar without accepting Boolean pseudo-numbers."""

    message = f"{field_name} must be a finite real scalar"
    scalar = _unwrapped_real_scalar(value, field_name=field_name)
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def _validated_config(
    config: _IMPL.TemporalConsensusConfig | None,
) -> _IMPL.TemporalConsensusConfig:
    """Resolve, validate, and normalize one temporal-consensus configuration."""

    if config is None:
        resolved = _IMPL.TemporalConsensusConfig()
    elif not isinstance(config, _IMPL.TemporalConsensusConfig):
        raise TypeError("config must be a TemporalConsensusConfig instance or None")
    else:
        resolved = config

    values = {
        field_name: _finite_real_scalar(
            getattr(resolved, field_name),
            field_name=field_name,
        )
        for field_name in _FINITE_CONFIG_FIELDS
    }
    for field_name in _POSITIVE_CONFIG_FIELDS:
        if values[field_name] <= 0.0:
            raise ValueError(f"{field_name} must be positive")
    return replace(resolved, **values)


def _validate_config(config: _IMPL.TemporalConsensusConfig) -> None:
    """Validate temporal-consensus settings at the legacy internal boundary."""

    _validated_config(config)


def add_temporal_candidate_consensus(
    candidates: Any,
    *,
    config: _IMPL.TemporalConsensusConfig | None = None,
) -> Any:
    """Attach temporal features after validating the explicit configuration."""

    return _ORIGINAL_ADD_TEMPORAL_CANDIDATE_CONSENSUS(
        candidates,
        config=_validated_config(config),
    )


_IMPL._validate_config = _validate_config
_IMPL.add_temporal_candidate_consensus = add_temporal_candidate_consensus

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_unwrapped_real_scalar"] = _unwrapped_real_scalar
globals()["_finite_real_scalar"] = _finite_real_scalar
globals()["_validated_config"] = _validated_config
globals()["_validate_config"] = _validate_config
globals()["add_temporal_candidate_consensus"] = add_temporal_candidate_consensus

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
