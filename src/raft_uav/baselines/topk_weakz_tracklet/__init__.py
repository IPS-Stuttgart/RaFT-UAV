"""Compatibility wrapper preserving exact Fortem track identifiers.

The maintained implementation lives in the sibling ``topk_weakz_tracklet.py``
module. This package keeps the public import path while preventing large,
fractional, Boolean, or malformed track identifiers from being rounded or
silently merged when tracklets are grouped.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from operator import index
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int as _optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "topk_weakz_tracklet.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._topk_weakz_tracklet_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load top-k weak-z tracklet implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_FORTEM_TRACKLETS = _IMPL.build_fortem_tracklets
_ORIGINAL_CONFIG_POST_INIT = _IMPL.TopKWeakZTrackletConfig.__post_init__

_POSITIVE_INTEGER_FIELDS = (
    "top_k_paths",
    "beam_width",
    "max_tracklets",
    "min_tracklet_length",
)
_POSITIVE_REAL_FIELDS = (
    "max_intra_tracklet_gap_s",
    "max_transition_gap_s",
    "max_transition_speed_mps",
    "max_transition_altitude_jump_m",
    "range_slack_m",
    "weakz_radar_xy_std_m",
    "weakz_radar_z_std_m",
    "acceleration_std_mps2",
    "smoother_lag_s",
    "smoother_acceleration_std_mps2",
    "rf_radar_consistency_std_m",
    "rf_min_reliability",
    "rf_max_covariance_scale",
    "rf_outside_radar_scale",
)
_NONNEGATIVE_REAL_FIELDS = (
    "track_switch_cost",
    "gap_cost_per_s",
    "speed_cost_weight",
    "altitude_jump_cost_weight",
    "tracklet_length_reward",
    "catprob_reward_weight",
    "confidence_reward_weight",
    "range_penalty_weight",
    "replay_nis_weight",
    "replay_rejection_penalty",
)
_OPTIONAL_POSITIVE_REAL_FIELDS = (
    "range_gate_m",
    "rf_reject_distance_m",
)


def _finite_real(value: object, *, name: str) -> float:
    message = f"{name} must be a finite real scalar"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(message)
    try:
        number = float(array.item())
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(number):
        raise ValueError(message)
    return number


def _positive_integer(value: object, *, name: str) -> int:
    message = f"{name} must be a positive integer"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        number = index(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if number < 1:
        raise ValueError(message)
    return int(number)


def _validated_config_post_init(config: object) -> None:
    """Reject malformed controls before they reach scoring or slicing code."""

    for name in _POSITIVE_INTEGER_FIELDS:
        _positive_integer(getattr(config, name), name=name)
    for name in _POSITIVE_REAL_FIELDS:
        if _finite_real(getattr(config, name), name=name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in _NONNEGATIVE_REAL_FIELDS:
        if _finite_real(getattr(config, name), name=name) < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    for name in _OPTIONAL_POSITIVE_REAL_FIELDS:
        value = getattr(config, name)
        if value is not None and _finite_real(value, name=name) <= 0.0:
            raise ValueError(f"{name} must be positive or None")
    _ORIGINAL_CONFIG_POST_INIT(config)


def build_fortem_tracklets(
    radar: pd.DataFrame,
    config: Any = None,
):
    """Build tracklets without routing integer identifiers through ``float``.

    The legacy implementation uses floating-point group keys. Dense surrogate
    integers preserve its grouping behavior while exact identifiers are restored
    on the returned immutable tracklet records. Invalid identifiers remain
    missing and therefore retain the legacy row-isolation behavior.
    """

    frame = pd.DataFrame(radar).copy()
    if frame.empty or "track_id" not in frame.columns:
        return _ORIGINAL_BUILD_FORTEM_TRACKLETS(frame, config)

    exact_ids = [_optional_int(value) for value in frame["track_id"].tolist()]
    surrogate_by_id: dict[int, int] = {}
    surrogates: list[float | int] = []
    for exact_id in exact_ids:
        if exact_id is None:
            surrogates.append(np.nan)
            continue
        surrogate = surrogate_by_id.setdefault(exact_id, len(surrogate_by_id))
        surrogates.append(surrogate)
    frame["track_id"] = surrogates

    tracklets = _ORIGINAL_BUILD_FORTEM_TRACKLETS(frame, config)
    restored = []
    for tracklet in tracklets:
        first_row = tracklet.row_indices[0]
        restored.append(replace(tracklet, track_id=exact_ids[first_row]))
    return restored


_IMPL.TopKWeakZTrackletConfig.__post_init__ = _validated_config_post_init
_IMPL._optional_int = _optional_int
_IMPL.build_fortem_tracklets = build_fortem_tracklets

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_optional_int"] = _optional_int
globals()["build_fortem_tracklets"] = build_fortem_tracklets
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
