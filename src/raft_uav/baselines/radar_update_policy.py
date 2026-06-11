"""Conservative radar-update policy helpers.

The policy is intentionally small and environment-driven so it can be enabled in
ablation and leave-flight-out runs without changing the default tracker path. It
uses only causal, truth-free quantities already written by the association
stages: association NIS, RF-anchor disagreement, candidate-mixture entropy,
effective candidate count, and recovery/miss-streak metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement, TrackingUpdateDiagnostics

ENV_DO_NO_HARM_RADAR_UPDATES = "RAFT_UAV_DO_NO_HARM_RADAR_UPDATES"
ENV_DO_NO_HARM_RADAR_UPDATE_POLICY = "RAFT_UAV_DO_NO_HARM_RADAR_UPDATE_POLICY"
ENV_DNH_SOFTEN_NIS = "RAFT_UAV_DNH_SOFTEN_NIS"
ENV_DNH_SKIP_NIS = "RAFT_UAV_DNH_SKIP_NIS"
ENV_DNH_ANCHOR_SOFTEN_NIS = "RAFT_UAV_DNH_ANCHOR_SOFTEN_NIS"
ENV_DNH_ANCHOR_SKIP_NIS = "RAFT_UAV_DNH_ANCHOR_SKIP_NIS"
ENV_DNH_ENTROPY_SOFTEN = "RAFT_UAV_DNH_ENTROPY_SOFTEN"
ENV_DNH_ENTROPY_DEFER = "RAFT_UAV_DNH_ENTROPY_DEFER"
ENV_DNH_EFFECTIVE_CANDIDATES_SOFTEN = "RAFT_UAV_DNH_EFFECTIVE_CANDIDATES_SOFTEN"
ENV_DNH_EFFECTIVE_CANDIDATES_DEFER = "RAFT_UAV_DNH_EFFECTIVE_CANDIDATES_DEFER"
ENV_DNH_RECOVERY_MISS_STREAK = "RAFT_UAV_DNH_RECOVERY_MISS_STREAK"
ENV_DNH_MAX_COVARIANCE_SCALE = "RAFT_UAV_DNH_MAX_COVARIANCE_SCALE"


@dataclass(frozen=True)
class RadarUpdatePolicy:
    """Thresholds for conservative radar update decisions."""

    soften_nis: float = 16.0
    skip_nis: float = 36.0
    anchor_soften_nis: float = 16.0
    anchor_skip_nis: float = 25.0
    entropy_soften: float = 1.0
    entropy_defer: float = 1.35
    effective_candidates_soften: float = 2.0
    effective_candidates_defer: float = 3.0
    recovery_miss_streak: int = 2
    max_covariance_scale: float = 25.0

    def __post_init__(self) -> None:
        if self.soften_nis <= 0.0 or self.skip_nis <= self.soften_nis:
            raise ValueError("skip_nis must be greater than positive soften_nis")
        if self.anchor_soften_nis <= 0.0 or self.anchor_skip_nis <= self.anchor_soften_nis:
            raise ValueError("anchor_skip_nis must be greater than positive anchor_soften_nis")
        if self.entropy_soften < 0.0 or self.entropy_defer < self.entropy_soften:
            raise ValueError("entropy_defer must be >= entropy_soften >= 0")
        if self.effective_candidates_soften < 1.0:
            raise ValueError("effective_candidates_soften must be >= 1")
        if self.effective_candidates_defer < self.effective_candidates_soften:
            raise ValueError("effective_candidates_defer must be >= effective_candidates_soften")
        if self.recovery_miss_streak < 1:
            raise ValueError("recovery_miss_streak must be positive")
        if self.max_covariance_scale < 1.0:
            raise ValueError("max_covariance_scale must be >= 1")


@dataclass(frozen=True)
class RadarUpdatePlan:
    """Decision returned by the do-no-harm policy."""

    action: str
    covariance_scale: float
    reason: str
    nis: float | None = None
    anchor_nis: float | None = None
    entropy: float | None = None
    effective_candidates: float | None = None
    preceding_miss_streak: int = 0

    @property
    def skipped(self) -> bool:
        return self.action in {"skip", "defer"}


def policy_from_environment() -> RadarUpdatePolicy | None:
    """Return a policy when the environment flag is enabled."""

    if not (
        _env_flag(ENV_DO_NO_HARM_RADAR_UPDATES)
        or _env_flag(ENV_DO_NO_HARM_RADAR_UPDATE_POLICY)
    ):
        return None
    return RadarUpdatePolicy(
        soften_nis=_env_float(ENV_DNH_SOFTEN_NIS, 16.0),
        skip_nis=_env_float(ENV_DNH_SKIP_NIS, 36.0),
        anchor_soften_nis=_env_float(ENV_DNH_ANCHOR_SOFTEN_NIS, 16.0),
        anchor_skip_nis=_env_float(ENV_DNH_ANCHOR_SKIP_NIS, 25.0),
        entropy_soften=_env_float(ENV_DNH_ENTROPY_SOFTEN, 1.0),
        entropy_defer=_env_float(ENV_DNH_ENTROPY_DEFER, 1.35),
        effective_candidates_soften=_env_float(ENV_DNH_EFFECTIVE_CANDIDATES_SOFTEN, 2.0),
        effective_candidates_defer=_env_float(ENV_DNH_EFFECTIVE_CANDIDATES_DEFER, 3.0),
        recovery_miss_streak=_env_int(ENV_DNH_RECOVERY_MISS_STREAK, 2),
        max_covariance_scale=_env_float(ENV_DNH_MAX_COVARIANCE_SCALE, 25.0),
    )


def classify_radar_update_row(
    row: pd.Series | dict[str, Any],
    policy: RadarUpdatePolicy | None = None,
) -> RadarUpdatePlan:
    """Classify a selected radar row as apply, soften, defer, or skip."""

    policy = policy or RadarUpdatePolicy()
    nis = _first_finite(row, "association_nis", "nis", "association_score")
    anchor_nis = _first_finite(row, "association_anchor_nis", "rf_anchor_nis")
    entropy = _first_finite(
        row,
        "association_soft_path_weight_entropy",
        "association_weight_entropy",
        "candidate_ambiguity_index",
    )
    effective = _effective_candidate_count(row, entropy)
    miss_streak = _finite_int(_get(row, "association_preceding_miss_streak")) or 0

    reasons: list[str] = []
    severe = False
    if nis is not None and nis >= policy.skip_nis:
        severe = True
        reasons.append(f"association_nis>={policy.skip_nis:g}")
    if anchor_nis is not None and anchor_nis >= policy.anchor_skip_nis:
        severe = True
        reasons.append(f"anchor_nis>={policy.anchor_skip_nis:g}")
    if entropy is not None and entropy >= policy.entropy_defer:
        severe = True
        reasons.append(f"entropy>={policy.entropy_defer:g}")
    if effective is not None and effective >= policy.effective_candidates_defer:
        severe = True
        reasons.append(f"effective_candidates>={policy.effective_candidates_defer:g}")
    if severe:
        action = "defer" if miss_streak >= policy.recovery_miss_streak else "skip"
        return RadarUpdatePlan(
            action=action,
            covariance_scale=1.0,
            reason=";".join(reasons),
            nis=nis,
            anchor_nis=anchor_nis,
            entropy=entropy,
            effective_candidates=effective,
            preceding_miss_streak=miss_streak,
        )

    scale_terms: list[float] = []
    if nis is not None and nis >= policy.soften_nis:
        scale_terms.append(nis / max(policy.soften_nis, 1.0e-9))
        reasons.append(f"association_nis>={policy.soften_nis:g}")
    if anchor_nis is not None and anchor_nis >= policy.anchor_soften_nis:
        scale_terms.append(anchor_nis / max(policy.anchor_soften_nis, 1.0e-9))
        reasons.append(f"anchor_nis>={policy.anchor_soften_nis:g}")
    if entropy is not None and entropy >= policy.entropy_soften:
        scale_terms.append(1.0 + entropy / max(policy.entropy_soften, 1.0e-9))
        reasons.append(f"entropy>={policy.entropy_soften:g}")
    if effective is not None and effective >= policy.effective_candidates_soften:
        scale_terms.append(effective / max(policy.effective_candidates_soften, 1.0e-9))
        reasons.append(f"effective_candidates>={policy.effective_candidates_soften:g}")

    if scale_terms:
        scale = float(np.clip(max(scale_terms), 1.0, policy.max_covariance_scale))
        return RadarUpdatePlan(
            action="soften",
            covariance_scale=scale,
            reason=";".join(reasons),
            nis=nis,
            anchor_nis=anchor_nis,
            entropy=entropy,
            effective_candidates=effective,
            preceding_miss_streak=miss_streak,
        )
    return RadarUpdatePlan(
        action="apply",
        covariance_scale=1.0,
        reason="nominal",
        nis=nis,
        anchor_nis=anchor_nis,
        entropy=entropy,
        effective_candidates=effective,
        preceding_miss_streak=miss_streak,
    )


def apply_radar_update_policy(
    row: pd.Series,
    measurement: TrackingMeasurement,
    policy: RadarUpdatePolicy | None = None,
) -> tuple[pd.Series, TrackingMeasurement, TrackingUpdateDiagnostics | None]:
    """Return an annotated row, possibly softened measurement, or skip diagnostics."""

    active_policy = policy or policy_from_environment()
    if active_policy is None:
        return row, measurement, None

    plan = classify_radar_update_row(row, active_policy)
    annotated = row.copy()
    annotated["association_update_policy"] = plan.action
    annotated["association_update_policy_reason"] = plan.reason
    annotated["association_update_policy_covariance_scale"] = float(plan.covariance_scale)
    annotated["association_update_policy_entropy"] = plan.entropy
    annotated["association_update_policy_effective_candidates"] = plan.effective_candidates

    if plan.skipped:
        return annotated, measurement, TrackingUpdateDiagnostics(
            time_s=float(measurement.time_s),
            source=measurement.source,
            measurement_dim=int(measurement.vector.size),
            accepted=False,
            update_action=f"do_no_harm_{plan.action}",
            nis=float("nan") if plan.nis is None else float(plan.nis),
            gate_threshold=None,
            safety_gate_threshold=None,
            residual_gate_threshold_m=None,
            covariance_scale=float(plan.covariance_scale),
            inflation_alpha=None,
            residual_norm_m=float("nan"),
        )
    if plan.action == "soften" and plan.covariance_scale > 1.0:
        measurement = _copy_measurement_with_covariance(
            measurement,
            np.asarray(measurement.covariance, dtype=float) * float(plan.covariance_scale),
        )
    return annotated, measurement, None


def policy_record_fields(row: pd.Series | dict[str, Any]) -> dict[str, object]:
    """Return row policy fields suitable for an estimate/diagnostic record."""

    fields = (
        "association_update_policy",
        "association_update_policy_reason",
        "association_update_policy_covariance_scale",
        "association_update_policy_entropy",
        "association_update_policy_effective_candidates",
    )
    out: dict[str, object] = {}
    for field in fields:
        value = _get(row, field)
        if value is not None and not _is_missing(value):
            out[field] = value
    return out


def _copy_measurement_with_covariance(
    measurement: TrackingMeasurement,
    covariance: np.ndarray,
) -> TrackingMeasurement:
    """Copy a measurement without re-applying source calibration side effects."""

    copied = object.__new__(TrackingMeasurement)
    object.__setattr__(copied, "time_s", float(measurement.time_s))
    object.__setattr__(copied, "vector", np.asarray(measurement.vector, dtype=float).copy())
    object.__setattr__(copied, "covariance", np.asarray(covariance, dtype=float).copy())
    object.__setattr__(copied, "source", str(measurement.source))
    return copied


def _effective_candidate_count(row: pd.Series | dict[str, Any], entropy: float | None) -> float | None:
    """Return an effective ambiguity count from explicit columns or entropy.

    Soft tracklet-Viterbi rows currently expose Shannon entropy as
    ``association_soft_path_weight_entropy``.  The effective number of equally
    plausible alternatives is ``exp(entropy)``.  Falling back to the raw soft
    path count is intentionally last because it is only an upper bound when path
    weights are strongly peaked.
    """

    explicit = _first_finite(
        row,
        "association_effective_candidates",
        "association_soft_path_effective_candidates",
    )
    if explicit is not None:
        return explicit
    if entropy is not None:
        return float(np.exp(float(entropy)))
    return _first_finite(row, "association_soft_path_count")


def _first_finite(row: pd.Series | dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = _finite_float(_get(row, name))
        if value is not None:
            return value
    return None


def _get(row: pd.Series | dict[str, Any], name: str) -> Any:
    if isinstance(row, pd.Series):
        return row.get(name)
    return row.get(name)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite_int(value: Any) -> int | None:
    number = _finite_float(value)
    return None if number is None else int(number)


def _is_missing(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return float(default)
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    return int(float(value))
