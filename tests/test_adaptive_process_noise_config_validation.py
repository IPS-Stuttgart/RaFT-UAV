from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.adaptive_process_noise import (
    ENV_ADAPTIVE_PROCESS_NOISE,
    AdaptiveProcessNoiseConfig,
    adaptive_process_noise_from_environment,
)


@pytest.mark.parametrize(
    "value",
    [np.nan, np.inf, -np.inf, True, np.array([4.0])],
)
def test_adaptive_process_noise_rejects_invalid_base_acceleration(value: object) -> None:
    with pytest.raises(
        ValueError,
        match="base_acceleration_std_mps2 must be a finite positive real scalar",
    ):
        AdaptiveProcessNoiseConfig(base_acceleration_std_mps2=value)


@pytest.mark.parametrize("value", [True, np.bool_(True), "4.0"])
def test_environment_factory_preserves_base_acceleration_validation(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    monkeypatch.setenv(ENV_ADAPTIVE_PROCESS_NOISE, "1")

    with pytest.raises(
        ValueError,
        match="base_acceleration_std_mps2 must be a finite positive real scalar",
    ):
        adaptive_process_noise_from_environment(base_acceleration_std_mps2=value)


def test_adaptive_process_noise_validates_threshold_order_at_config_boundary() -> None:
    with pytest.raises(
        ValueError,
        match="high_nis_ratio must be at least low_nis_ratio",
    ):
        AdaptiveProcessNoiseConfig(low_nis_ratio=0.8, high_nis_ratio=0.5)


def test_adaptive_process_noise_normalizes_scalar_like_values() -> None:
    config = AdaptiveProcessNoiseConfig(
        base_acceleration_std_mps2=np.array(4.0),
        min_scale=np.float64(0.5),
        max_scale=np.array(3.0),
    )

    assert config.base_acceleration_std_mps2 == 4.0
    assert config.min_scale == 0.5
    assert config.max_scale == 3.0
    assert isinstance(config.base_acceleration_std_mps2, float)
    assert isinstance(config.min_scale, float)
    assert isinstance(config.max_scale, float)
