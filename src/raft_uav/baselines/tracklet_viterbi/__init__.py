"""Compatibility validation for tracklet Viterbi association.

The maintained implementation remains in the sibling ``tracklet_viterbi.py``
module. This package preserves the public import path while rejecting non-finite
numeric configuration controls before they can enter path-cost calculations.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
from typing import Any

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_viterbi.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._tracklet_viterbi_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load tracklet Viterbi implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_CONFIG_POST_INIT = _IMPL.TrackletViterbiAssociationConfig.__post_init__
_NUMERIC_CONFIG_FIELDS = (
    "max_candidates_per_frame",
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
    "range_gate_m",
    "range_gate_slack_m",
    "range_penalty",
    "reacquisition_miss_streak_threshold",
    "reacquisition_gate_nis",
    "reacquisition_gate_growth",
    "reacquisition_reward",
    "reacquisition_outside_gate_penalty",
    "min_learned_candidate_probability",
    "min_catprob",
    "soft_top_k_paths",
    "soft_path_temperature",
)


def _validate_config_with_finite_controls(config: Any) -> None:
    """Reject NaN and infinite controls before existing range validation."""

    for name in _NUMERIC_CONFIG_FIELDS:
        value = getattr(config, name)
        if name == "range_gate_m" and value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite")
    _ORIGINAL_CONFIG_POST_INIT(config)


_IMPL.TrackletViterbiAssociationConfig.__post_init__ = (
    _validate_config_with_finite_controls
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)

__doc__ = _IMPL.__doc__
__all__ = [
    name
    for name in dir(_IMPL)
    if not (name.startswith("__") and name.endswith("__"))
]
