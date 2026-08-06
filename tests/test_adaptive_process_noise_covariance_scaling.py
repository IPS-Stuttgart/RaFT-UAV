from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.adaptive_process_noise import (
    AdaptiveProcessNoiseConfig,
    RollingNISAdaptiveAcceleration,
)
from raft_uav.baselines.kalman import white_acceleration_process_noise


@pytest.mark.parametrize(
    ("nis", "expected_covariance_scale"),
    [(12.0, 2.25), (0.0, 0.7)],
)
def test_adaptive_acceleration_preserves_covariance_multiplier(
    nis: float,
    expected_covariance_scale: float,
) -> None:
    base_std_mps2 = 4.0
    adapter = RollingNISAdaptiveAcceleration(
        AdaptiveProcessNoiseConfig(
            base_acceleration_std_mps2=base_std_mps2,
            ewma_alpha=1.0,
        )
    )
    adapter.observe(source="radar", measurement_dim=3, nis=nis)

    nominal_covariance = white_acceleration_process_noise(
        dt_s=0.5,
        acceleration_std=base_std_mps2,
    )
    adapted_covariance = white_acceleration_process_noise(
        dt_s=0.5,
        acceleration_std=adapter.acceleration_std_mps2(),
    )

    assert np.allclose(
        adapted_covariance,
        nominal_covariance * expected_covariance_scale,
    )
