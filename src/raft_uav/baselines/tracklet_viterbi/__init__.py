"""Compatibility package validating tracklet-Viterbi numeric controls.

The maintained implementation lives in the sibling ``tracklet_viterbi.py``
module. This package preserves the public import path while rejecting malformed
numeric configuration before it can enter association costs and gates. It also
keeps soft top-K path mixtures from turning minority-path detections into radar
updates when the path posterior favors a missed detection.
"""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Any

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
_ORIGINAL_REACQUISITION_COST = _IMPL._reacquisition_cost
_ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_TRACKLET_VITERBI_ASSOCIATION = (
    _IMPL.run_async_cv_baseline_with_tracklet_viterbi_association
)
_ORIGINAL_SELECTED_ROWS_FROM_SOFT_VITERBI_PATHS = (
    _IMPL._selected_rows_from_soft_viterbi_paths
)

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


def _validate_association_config(config: Any) -> None:
    """Reject malformed explicit configs instead of replacing falsy values."""

    if config is not None and not isinstance(
        config,
        _IMPL.TrackletViterbiAssociationConfig,
    ):
        raise ValueError(
            "config must be a TrackletViterbiAssociationConfig instance or None"
        )


@wraps(_ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_TRACKLET_VITERBI_ASSOCIATION)
def run_async_cv_baseline_with_tracklet_viterbi_association(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run tracklet-Viterbi association after validating an explicit config."""

    if not args:
        _validate_association_config(kwargs.get("config"))
    return _ORIGINAL_RUN_ASYNC_CV_BASELINE_WITH_TRACKLET_VITERBI_ASSOCIATION(
        *args,
        **kwargs,
    )


def _bounded_reacquisition_cost(
    previous_miss_streak: int,
    current: Any,
    config: Any,
    *,
    has_prior_detection: bool = True,
) -> float:
    """Prevent reacquisition rewards from making extra misses profitable."""

    cost = float(
        _ORIGINAL_REACQUISITION_COST(
            previous_miss_streak,
            current,
            config,
            has_prior_detection=has_prior_detection,
        )
    )
    return max(cost, -float(config.missed_detection_cost))


def _miss_aware_selected_rows_from_soft_viterbi_paths(
    paths: list[tuple[float, list[Any]]],
    config: Any,
) -> list[Any]:
    """Keep a radar update only when detection path mass wins over miss mass.

    The legacy soft-path combiner conditions on paths that contain a detection
    and therefore drops all missed-detection probability. A single low-weight
    path can consequently create a radar update at a frame where the dominant
    path and almost all posterior mass choose the miss node. Compute path
    weights once over the complete top-K set, use marginal majority mass for
    the discrete detection/miss decision, and retain the legacy conditional
    moment matching only for frames whose detection state is selected.
    """

    if not paths:
        return []

    path_costs = np.asarray([float(path_cost) for path_cost, _ in paths], dtype=float)
    path_weights = _IMPL._soft_path_weights(path_costs, config)
    best_path_index = int(np.argmax(path_weights))

    detection_probability: dict[tuple[str, int | float], float] = {}
    detection_path_count: dict[tuple[str, int | float], int] = {}
    best_path_detection_keys: set[tuple[str, int | float]] = set()

    for path_index, (_, path) in enumerate(paths):
        seen_keys: set[tuple[str, int | float]] = set()
        for node in path:
            key = node.event_key
            if key in seen_keys:
                continue
            seen_keys.add(key)
            detection_probability.setdefault(key, 0.0)
            detection_path_count.setdefault(key, 0)
            if node.is_miss or node.row is None:
                continue
            detection_probability[key] += float(path_weights[path_index])
            detection_path_count[key] += 1
            if path_index == best_path_index:
                best_path_detection_keys.add(key)

    selected_rows = _ORIGINAL_SELECTED_ROWS_FROM_SOFT_VITERBI_PATHS(paths, config)
    retained: list[Any] = []
    for selected_row in selected_rows:
        key = _IMPL._selected_row_event_key(selected_row)
        probability = float(detection_probability.get(key, 0.0))
        is_tie = bool(np.isclose(probability, 0.5, rtol=0.0, atol=1.0e-12))
        if probability < 0.5 and not is_tie:
            continue
        if is_tie and key not in best_path_detection_keys:
            continue

        row = selected_row.copy()
        row["association_soft_path_count"] = int(len(paths))
        row["association_soft_detection_path_count"] = int(
            detection_path_count.get(key, 0)
        )
        row["association_soft_detection_probability"] = probability
        row["association_soft_miss_probability"] = float(
            np.clip(1.0 - probability, 0.0, 1.0)
        )
        retained.append(row)
    return retained


_IMPL.TrackletViterbiAssociationConfig.__post_init__ = _validated_config_post_init
_IMPL.run_async_cv_baseline_with_tracklet_viterbi_association = (
    run_async_cv_baseline_with_tracklet_viterbi_association
)
_IMPL._reacquisition_cost = _bounded_reacquisition_cost
_IMPL._selected_rows_from_soft_viterbi_paths = (
    _miss_aware_selected_rows_from_soft_viterbi_paths
)

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_config_post_init"] = _validated_config_post_init
globals()["_validate_association_config"] = _validate_association_config
globals()["run_async_cv_baseline_with_tracklet_viterbi_association"] = (
    run_async_cv_baseline_with_tracklet_viterbi_association
)
globals()["_bounded_reacquisition_cost"] = _bounded_reacquisition_cost
globals()["_miss_aware_selected_rows_from_soft_viterbi_paths"] = (
    _miss_aware_selected_rows_from_soft_viterbi_paths
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
