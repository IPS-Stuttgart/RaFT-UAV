from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.pyrecest_measurement_reliability import (
    apply_raft_measurement_reliability,
    rf_reliability_covariance_scale,
    scale_rf_covariance_by_reliability,
)


def test_soft_mode_inflates_covariance_via_pyrecest() -> None:
    decision = apply_raft_measurement_reliability(
        np.eye(2),
        reliability=0.5,
        mode="soft",
        min_probability=0.05,
    )

    assert decision.accepted
    assert decision.update_action == "rf_reliability_scaled"
    assert decision.covariance_scale == 2.0
    assert np.allclose(decision.covariance, 2.0 * np.eye(2))


def test_hard_mode_rejects_below_threshold_without_scaling_covariance() -> None:
    decision = apply_raft_measurement_reliability(
        np.eye(2),
        reliability=0.4,
        mode="hard",
        threshold=0.45,
    )

    assert not decision.accepted
    assert decision.update_action == "rf_reliability_rejected"
    assert decision.covariance_scale == 1.0
    assert np.allclose(decision.covariance, np.eye(2))


def test_off_mode_preserves_nominal_covariance() -> None:
    covariance = np.diag([9.0, 16.0])
    decision = apply_raft_measurement_reliability(covariance, reliability=0.1, mode="off")

    assert decision.accepted
    assert decision.update_action == "rf_reliability_off"
    assert decision.covariance_scale == 1.0
    assert np.allclose(decision.covariance, covariance)


def test_legacy_scale_helpers_delegate_to_pyrecest() -> None:
    covariance = np.eye(2)
    scaled, scale = scale_rf_covariance_by_reliability(covariance, 0.25)

    assert rf_reliability_covariance_scale(0.25) == 4.0
    assert scale == 4.0
    assert np.allclose(scaled, 4.0 * covariance)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        apply_raft_measurement_reliability(np.eye(2), reliability=0.5, mode="unknown")
