"""Compatibility validation for MMUAD temporal-consensus configuration.

The maintained implementation lives in the sibling
``candidate_temporal_consensus.py`` module. This package preserves its public
import path while rejecting non-finite numeric settings before they can disable
gating or propagate non-finite consensus scores.
"""

from __future__ import annotations

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
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load temporal-consensus implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_VALIDATE_CONFIG = _IMPL._validate_config
_FINITE_CONFIG_FIELDS = (
    "max_time_gap_s",
    "max_speed_mps",
    "distance_scale_m",
    "acceleration_scale_mps2",
    "base_score_weight",
    "backward_support_weight",
    "forward_support_weight",
    "bidirectional_bonus",
    "interpolation_weight",
    "acceleration_weight",
    "source_diversity_bonus",
    "branch_diversity_bonus",
)


def _validate_config(config: Any) -> None:
    """Reject NaN and infinite temporal-consensus settings."""

    for field_name in _FINITE_CONFIG_FIELDS:
        try:
            value = float(getattr(config, field_name))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"{field_name} must be finite")
    _ORIGINAL_VALIDATE_CONFIG(config)


_IMPL._validate_config = _validate_config

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validate_config"] = _validate_config
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
