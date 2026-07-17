"""Compatibility package validating tracklet-Viterbi numeric controls.

The maintained implementation lives in the sibling ``tracklet_viterbi.py``
module. This package preserves the public import path while rejecting malformed
numeric configuration before it can enter association costs and gates.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.numeric import optional_float, optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_viterbi.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._tracklet_viterbi_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load tracklet-Viterbi implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_CONFIG_POST_INIT = _IMPL.TrackletViterbiAssociationConfig.__post_init__
_ORIGINAL_REACQUISITION_COST = _IMPL._reacquisition_cost

_INTEGER_CONFIG_FIELDS = (
    "max_candidates_per_frame",
    "reacquisition_miss_streak_threshold",
    "soft_top_k_paths",
)
_FLOAT_CONFIG_FIELDS = (
    "missed_detection_cost",
    "consecutive_miss_cost",
    "track_switch_cost",
    "missing_track_id_cost",
    "catprob_weight",
    "anchor_nis_weight",
    "transition_nis_weight",
    "velocity_nis_weight",
    "transition_position_std_m",
    "transition_speed_std_mps",
    "velocity_std_mps",
    "max_speed_mps",
    "max_speed_penalty",
    "range_gate_slack_m",
    "range_penalty",
    "reacquisition_gate_nis",
    "reacquisition_gate_growth",
    "reacquisition_reward",
    "reacquisition_outside_gate_penalty",
    "min_learned_candidate_probability",
    "min_catprob",
    "soft_path_temperature",
)


def _validated_config_post_init(self: Any) -> None:
    """Normalize finite scalar controls before applying domain constraints."""

    for name in _INTEGER_CONFIG_FIELDS:
        value = optional_int(getattr(self, name))
        if value is None:
            raise ValueError(f"{name} must be a finite integer scalar")
        object.__setattr__(self, name, value)

    for name in _FLOAT_CONFIG_FIELDS:
        value = optional_float(getattr(self, name))
        if value is None:
            raise ValueError(f"{name} must be a finite real scalar")
        object.__setattr__(self, name, value)

    if self.range_gate_m is not None:
        range_gate_m = optional_float(self.range_gate_m)
        if range_gate_m is None:
            raise ValueError("range_gate_m must be a finite real scalar or None")
        object.__setattr__(self, "range_gate_m", range_gate_m)

    _ORIGINAL_CONFIG_POST_INIT(self)


def _bounded_reacquisition_cost(
    previous_miss_streak: int,
    current: Any,
    config: Any,
) -> float:
    """Prevent reacquisition rewards from making extra misses profitable."""

    cost = float(_ORIGINAL_REACQUISITION_COST(previous_miss_streak, current, config))
    return max(cost, -float(config.missed_detection_cost))


_IMPL.TrackletViterbiAssociationConfig.__post_init__ = _validated_config_post_init
_IMPL._reacquisition_cost = _bounded_reacquisition_cost

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_config_post_init"] = _validated_config_post_init
globals()["_bounded_reacquisition_cost"] = _bounded_reacquisition_cost

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
