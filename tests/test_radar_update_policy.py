"""Tests for conservative radar update policy feature wiring."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.baselines.radar_update_policy import (
    ENV_DNH_RECOVERY_MISS_STREAK,
    ENV_DNH_SOFTEN_NIS,
    ENV_DO_NO_HARM_RADAR_UPDATE_POLICY,
    ENV_DO_NO_HARM_RADAR_UPDATES,
    RadarUpdatePolicy,
    classify_radar_update_row,
    policy_record_fields,
    policy_from_environment,
)


def test_do_no_harm_uses_soft_path_entropy_as_effective_candidate_count() -> None:
    row = pd.Series(
        {
            "association_nis": 1.0,
            "association_anchor_nis": 1.0,
            "association_soft_path_weight_entropy": math.log(3.2),
            "association_preceding_miss_streak": 0,
        }
    )

    plan = classify_radar_update_row(
        row,
        RadarUpdatePolicy(
            entropy_soften=10.0,
            entropy_defer=11.0,
            effective_candidates_soften=2.0,
            effective_candidates_defer=4.0,
        ),
    )

    assert plan.action == "soften"
    assert plan.entropy == math.log(3.2)
    assert plan.effective_candidates is not None
    assert plan.effective_candidates == pytest.approx(3.2)
    assert "effective_candidates>=2" in plan.reason


def test_do_no_harm_defers_severe_soft_path_entropy_after_recovery_streak() -> None:
    row = pd.Series(
        {
            "association_nis": 1.0,
            "association_anchor_nis": 1.0,
            "association_soft_path_weight_entropy": 1.5,
            "association_preceding_miss_streak": 2,
        }
    )

    plan = classify_radar_update_row(
        row,
        RadarUpdatePolicy(entropy_soften=1.0, entropy_defer=1.35, recovery_miss_streak=2),
    )

    assert plan.action == "defer"
    assert "entropy>=1.35" in plan.reason


def test_policy_record_fields_include_ambiguity_diagnostics() -> None:
    fields = policy_record_fields(
        {
            "association_update_policy": "soften",
            "association_update_policy_reason": "entropy>=1",
            "association_update_policy_covariance_scale": 2.0,
            "association_update_policy_entropy": 1.2,
            "association_update_policy_effective_candidates": 3.3,
        }
    )

    assert fields["association_update_policy_entropy"] == 1.2
    assert fields["association_update_policy_effective_candidates"] == 3.3


def test_policy_from_environment_accepts_compatibility_enable_alias(monkeypatch) -> None:
    """Legacy SOTA manifests should still enable do-no-harm radar updates."""

    monkeypatch.delenv(ENV_DO_NO_HARM_RADAR_UPDATES, raising=False)
    monkeypatch.setenv(ENV_DO_NO_HARM_RADAR_UPDATE_POLICY, "1")

    policy = policy_from_environment()

    assert isinstance(policy, RadarUpdatePolicy)


def test_radar_update_policy_rejects_nonfinite_thresholds() -> None:
    with pytest.raises(ValueError, match="soften_nis must be a finite number"):
        RadarUpdatePolicy(soften_nis=float("nan"))
    with pytest.raises(ValueError, match="max_covariance_scale must be a finite number"):
        RadarUpdatePolicy(max_covariance_scale=float("inf"))


def test_policy_from_environment_rejects_nonfinite_thresholds(monkeypatch) -> None:
    monkeypatch.setenv(ENV_DO_NO_HARM_RADAR_UPDATE_POLICY, "1")
    monkeypatch.setenv(ENV_DNH_SOFTEN_NIS, "nan")

    with pytest.raises(ValueError, match=ENV_DNH_SOFTEN_NIS):
        policy_from_environment()


def test_policy_from_environment_rejects_fractional_recovery_streak(monkeypatch) -> None:
    monkeypatch.setenv(ENV_DO_NO_HARM_RADAR_UPDATE_POLICY, "1")
    monkeypatch.setenv(ENV_DNH_RECOVERY_MISS_STREAK, "2.5")

    with pytest.raises(ValueError, match=ENV_DNH_RECOVERY_MISS_STREAK):
        policy_from_environment()
