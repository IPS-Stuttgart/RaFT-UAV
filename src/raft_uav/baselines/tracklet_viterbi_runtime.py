"""Runtime integration for first-class tracklet-Viterbi association."""

from __future__ import annotations

import os
from typing import Any

from raft_uav.baselines.tracklet_viterbi_retention import (
    TrackletViterbiAssociationConfig,
    run_async_cv_baseline_with_tracklet_viterbi_association,
)

_TRACKLET_MODE = "tracklet-viterbi"
_INSTALLED = False
_ORIGINAL_RUNNER: Any = None
_TRACKLET_RUNNER_KEYS = {
    "rf_measurements",
    "radar",
    "acceleration_std_mps2",
    "radar_xy_std_m",
    "radar_z_std_m",
    "gate_probabilities_by_source",
    "gate_thresholds_by_source",
    "safety_gate_probabilities_by_source",
    "safety_gate_thresholds_by_source",
    "robust_update_by_source",
    "inflation_alpha_by_source",
    "max_residual_norms_by_source",
    "candidate_catprob_threshold",
    "radar_covariance_fn",
}

_TRACKLET_CATPROB_RETENTION_MODES = ("hard", "soft")
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


class _TrackletConfigOverlay:
    """Expose base Viterbi config plus retention-only runtime fields."""

    def __init__(self, base: TrackletViterbiAssociationConfig, **overrides: object) -> None:
        self._base = base
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._base, name)


def install() -> None:
    """Register tracklet-Viterbi as a normal radar association mode."""

    global _INSTALLED, _ORIGINAL_RUNNER
    if _INSTALLED:
        return

    from raft_uav.baselines import radar_association

    _ORIGINAL_RUNNER = radar_association.run_async_cv_baseline_with_radar_association
    if _TRACKLET_MODE not in radar_association.RADAR_ASSOCIATION_MODES:
        radar_association.RADAR_ASSOCIATION_MODES = (
            *radar_association.RADAR_ASSOCIATION_MODES,
            _TRACKLET_MODE,
        )
    radar_association.run_async_cv_baseline_with_radar_association = (
        _run_async_cv_baseline_with_tracklet_dispatch
    )
    _INSTALLED = True


def _run_async_cv_baseline_with_tracklet_dispatch(
    **kwargs: Any,
) -> tuple[list[dict[str, object]], Any]:
    """Dispatch standard association modes or the tracklet-Viterbi runner."""

    association = kwargs.get("association")
    if association == _TRACKLET_MODE:
        tracklet_kwargs = {key: kwargs[key] for key in _TRACKLET_RUNNER_KEYS if key in kwargs}
        tracklet_kwargs["rf_measurements"] = list(tracklet_kwargs["rf_measurements"])
        tracklet_kwargs["config"] = _config_from_environment()
        return run_async_cv_baseline_with_tracklet_viterbi_association(**tracklet_kwargs)

    if _ORIGINAL_RUNNER is None:
        raise RuntimeError("tracklet-viterbi runtime was not installed correctly")
    return _ORIGINAL_RUNNER(**kwargs)


def _config_from_environment() -> _TrackletConfigOverlay:
    """Read optional ``RAFT_UAV_TRACKLET_*`` tuning values."""

    defaults = TrackletViterbiAssociationConfig()
    range_gate = _env_optional_positive_float(
        "RAFT_UAV_TRACKLET_RANGE_GATE_M",
        0.0 if defaults.range_gate_m is None else float(defaults.range_gate_m),
    )
    base = TrackletViterbiAssociationConfig(
        max_candidates_per_frame=_env_positive_int_any(
            (
                "RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_FRAME",
                "RAFT_UAV_TRACKLET_MAX_CANDIDATES",
            ),
            defaults.max_candidates_per_frame,
        ),
        missed_detection_cost=_env_positive_float(
            "RAFT_UAV_TRACKLET_MISSED_DETECTION_COST",
            float(defaults.missed_detection_cost),
        ),
        consecutive_miss_cost=_env_positive_float(
            "RAFT_UAV_TRACKLET_CONSECUTIVE_MISS_COST",
            float(defaults.consecutive_miss_cost),
        ),
        track_switch_cost=_env_positive_float(
            "RAFT_UAV_TRACKLET_TRACK_SWITCH_COST",
            float(defaults.track_switch_cost),
        ),
        missing_track_id_cost=_env_positive_float(
            "RAFT_UAV_TRACKLET_MISSING_TRACK_ID_COST",
            float(defaults.missing_track_id_cost),
        ),
        catprob_weight=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_CATPROB_WEIGHT",
            float(defaults.catprob_weight),
        ),
        anchor_nis_weight=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_ANCHOR_NIS_WEIGHT",
            float(defaults.anchor_nis_weight),
        ),
        transition_nis_weight=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_TRANSITION_NIS_WEIGHT",
            float(defaults.transition_nis_weight),
        ),
        velocity_nis_weight=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_VELOCITY_NIS_WEIGHT",
            float(defaults.velocity_nis_weight),
        ),
        transition_position_std_m=_env_positive_float(
            "RAFT_UAV_TRACKLET_TRANSITION_POSITION_STD_M",
            float(defaults.transition_position_std_m),
        ),
        transition_speed_std_mps=_env_positive_float(
            "RAFT_UAV_TRACKLET_TRANSITION_SPEED_STD_MPS",
            float(defaults.transition_speed_std_mps),
        ),
        velocity_std_mps=_env_positive_float(
            "RAFT_UAV_TRACKLET_VELOCITY_STD_MPS",
            float(defaults.velocity_std_mps),
        ),
        max_speed_mps=_env_positive_float(
            "RAFT_UAV_TRACKLET_MAX_SPEED_MPS",
            float(defaults.max_speed_mps),
        ),
        max_speed_penalty=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_MAX_SPEED_PENALTY",
            float(defaults.max_speed_penalty),
        ),
        range_gate_m=range_gate,
        range_gate_slack_m=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_RANGE_GATE_SLACK_M",
            float(defaults.range_gate_slack_m),
        ),
        range_penalty=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_RANGE_PENALTY",
            float(defaults.range_penalty),
        ),
        use_rf_anchor=_env_bool(
            "RAFT_UAV_TRACKLET_USE_RF_ANCHOR",
            bool(defaults.use_rf_anchor),
        ),
    )
    return _TrackletConfigOverlay(
        base,
        catprob_retention_mode=_env_choice(
            "RAFT_UAV_TRACKLET_CATPROB_RETENTION_MODE",
            "soft",
            _TRACKLET_CATPROB_RETENTION_MODES,
        ),
        below_catprob_threshold_penalty=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_BELOW_CATPROB_THRESHOLD_PENALTY",
            3.0,
        ),
        track_support_weight=_env_nonnegative_float("RAFT_UAV_TRACKLET_SUPPORT_WEIGHT", 0.45),
        max_track_support_reward=_env_nonnegative_float(
            "RAFT_UAV_TRACKLET_MAX_SUPPORT_REWARD",
            4.0,
        ),
        max_candidate_pool_per_frame=_env_positive_int(
            "RAFT_UAV_TRACKLET_MAX_CANDIDATE_POOL_PER_FRAME",
            24,
        ),
        max_candidates_per_track_id=_env_nonnegative_int(
            "RAFT_UAV_TRACKLET_MAX_CANDIDATES_PER_TRACK_ID",
            1,
        ),
    )


def _env_float(name: str, default: float) -> float:
    return _env_float_any((name,), default)


def _env_float_any(names: tuple[str, ...], default: float) -> float:
    return _env_number_any(names, default)[1]


def _env_positive_float(name: str, default: float) -> float:
    number = _env_float(name, default)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _env_optional_positive_float(name: str, default: float) -> float | None:
    number = _env_float(name, default)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return None if number == 0.0 else number


def _env_nonnegative_float(name: str, default: float) -> float:
    number = _env_float(name, default)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _env_positive_int(name: str, default: int) -> int:
    return _env_positive_int_any((name,), default)


def _env_positive_int_any(names: tuple[str, ...], default: int) -> int:
    env_name, number = _env_number_any(names, float(default))
    parsed = _integer_from_number(number, env_name)
    if parsed < 1:
        raise ValueError(f"{env_name} must be positive")
    return parsed


def _env_nonnegative_int(name: str, default: int) -> int:
    env_name, number = _env_number_any((name,), float(default))
    parsed = _integer_from_number(number, env_name)
    if parsed < 0:
        raise ValueError(f"{env_name} must be nonnegative")
    return parsed


def _integer_from_number(number: float, name: str) -> int:
    if not number.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(number)


def _env_number_any(names: tuple[str, ...], default: float) -> tuple[str, float]:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be numeric") from exc
        if not _is_finite(number):
            raise ValueError(f"{name} must be finite")
        return name, number
    default_number = float(default)
    if not _is_finite(default_number):
        raise ValueError(f"{names[0]} default must be finite")
    return names[0], default_number


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower()


def _env_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    parsed = _env_str(name, default)
    if parsed not in choices:
        raise ValueError(f"{name} must be one of {choices}")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    parsed = value.strip().lower()
    if parsed in _FALSE_ENV_VALUES:
        return False
    if parsed in _TRUE_ENV_VALUES:
        return True
    raise ValueError(f"{name} must be boolean")


def _is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
