"""Compatibility package hardening tracklet-Viterbi configuration and scoring.

The maintained implementation lives in the sibling ``tracklet_viterbi.py``
module. This package preserves the public import path while rejecting malformed
numeric configuration and preventing leading gaps from receiving a
reacquisition reward before any radar candidate has been acquired.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

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
_ORIGINAL_TRANSITION_COST = _IMPL._transition_cost
_ORIGINAL_SELECTED_ROWS_FROM_PATH = _IMPL._selected_rows_from_viterbi_path

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


def _is_leading_miss_streak(context: Any, previous: Any) -> bool:
    """Return whether every frame before the current transition was a gap."""

    if not bool(previous.is_missed_detection):
        return False
    frame_position = getattr(context, "frame_index", None)
    if frame_position is None:
        return False
    return int(context.previous_miss_streak) == int(frame_position)


def _top_k_viterbi_paths_without_leading_reacquisition(
    frames: list[list[Any]],
    config: Any,
    terminal_count: int,
) -> list[tuple[float, list[Any]]]:
    """Solve paths without rewarding the first-ever detection as reacquisition."""

    sequence_frames = [
        _IMPL._sequence_candidates_for_frame(frame_position, frame, config)
        for frame_position, frame in enumerate(frames)
    ]

    def transition_cost(previous: Any, current: Any, context: Any) -> float:
        previous_miss_streak = int(context.previous_miss_streak)
        if _is_leading_miss_streak(context, previous):
            previous_miss_streak = 0
        return _ORIGINAL_TRANSITION_COST(
            _IMPL._node_from_sequence_candidate(previous),
            _IMPL._node_from_sequence_candidate(current),
            config,
            previous_miss_streak=previous_miss_streak,
        )

    paths = _IMPL.solve_top_k_viterbi_sequence_associations(
        sequence_frames,
        transition_cost,
        top_k_terminal_paths=terminal_count,
    )
    return [
        (
            float(path.total_cost),
            [_IMPL._node_from_sequence_candidate(node) for node in path.nodes],
        )
        for path in paths
    ]


def _selected_rows_without_leading_reacquisition(
    path: Iterable[Any],
    path_cost: float,
    config: Any,
) -> list[Any]:
    """Keep leading-gap diagnostics distinct from genuine reacquisition."""

    nodes = list(path)
    rows = _ORIGINAL_SELECTED_ROWS_FROM_PATH(nodes, path_cost, config)
    leading_misses = 0
    for node in nodes:
        if node.is_miss or node.row is None:
            leading_misses += 1
            continue
        if leading_misses and rows:
            rows[0]["association_reacquisition_active"] = False
            rows[0]["association_reacquisition_cost"] = 0.0
            rows[0]["association_reacquisition_gate_nis"] = np.nan
        break
    return rows


_IMPL.TrackletViterbiAssociationConfig.__post_init__ = _validated_config_post_init
_IMPL._top_k_viterbi_paths_with_pyrecest = (
    _top_k_viterbi_paths_without_leading_reacquisition
)
_IMPL._selected_rows_from_viterbi_path = _selected_rows_without_leading_reacquisition

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_config_post_init"] = _validated_config_post_init
globals()["_is_leading_miss_streak"] = _is_leading_miss_streak
globals()["_top_k_viterbi_paths_without_leading_reacquisition"] = (
    _top_k_viterbi_paths_without_leading_reacquisition
)
globals()["_selected_rows_without_leading_reacquisition"] = (
    _selected_rows_without_leading_reacquisition
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
