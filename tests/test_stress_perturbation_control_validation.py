from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.stress.perturbations import (
    PerturbationConfig,
    corrupt_velocity,
    drop_radar_frames,
    drop_rf_bursts,
    inject_false_tracks,
    jitter_timestamps,
)


@pytest.mark.parametrize(
    "field",
    [
        "radar_drop_rate",
        "rf_drop_burst_rate",
        "timestamp_jitter_std_s",
        "false_track_position_std_m",
        "velocity_noise_std_mps",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.0])
def test_perturbation_config_rejects_invalid_nonnegative_controls(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite and nonnegative"):
        PerturbationConfig(name="invalid", **{field: value})


@pytest.mark.parametrize("field", ["radar_drop_rate", "rf_drop_burst_rate"])
@pytest.mark.parametrize("value", [1.000001, 2.0, "1.5"])
def test_perturbation_config_rejects_drop_rates_above_one(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must not exceed 1"):
        PerturbationConfig(name="invalid", **{field: value})


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_perturbation_config_rejects_invalid_covariance_scale(value: float) -> None:
    with pytest.raises(ValueError, match="covariance_scale must be finite and positive"):
        PerturbationConfig(name="invalid", covariance_scale=value)


@pytest.mark.parametrize("field", ["false_tracks_per_frame", "seed"])
@pytest.mark.parametrize("value", [-1, 1.5, True, np.nan, np.array([1])])
def test_perturbation_config_rejects_invalid_integer_controls(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be a nonnegative integer"):
        PerturbationConfig(name="invalid", **{field: value})


def test_perturbation_config_normalizes_numeric_string_controls() -> None:
    config = PerturbationConfig.from_mapping(
        {
            "name": "serialized",
            "radar_drop_rate": "0.25",
            "timestamp_jitter_std_s": "0.5",
            "false_tracks_per_frame": "2",
            "covariance_scale": "1.5",
            "seed": "17",
        }
    )

    assert config.radar_drop_rate == pytest.approx(0.25)
    assert config.timestamp_jitter_std_s == pytest.approx(0.5)
    assert config.false_tracks_per_frame == 2
    assert config.covariance_scale == pytest.approx(1.5)
    assert config.seed == 17


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda rng: drop_radar_frames(pd.DataFrame(), rate=np.nan, rng=rng),
            "rate must be finite and nonnegative",
        ),
        (
            lambda rng: drop_rf_bursts(pd.DataFrame(), rate=np.inf, rng=rng),
            "rate must be finite and nonnegative",
        ),
        (
            lambda rng: jitter_timestamps(pd.DataFrame(), std_s=np.nan, rng=rng),
            "std_s must be finite and nonnegative",
        ),
        (
            lambda rng: corrupt_velocity(pd.DataFrame(), std_mps=np.inf, rng=rng),
            "std_mps must be finite and nonnegative",
        ),
        (
            lambda rng: inject_false_tracks(
                pd.DataFrame(),
                false_tracks_per_frame=1.5,
                position_std_m=10.0,
                rng=rng,
            ),
            "false_tracks_per_frame must be a nonnegative integer",
        ),
    ],
)
def test_public_helpers_reject_invalid_controls_before_empty_return(
    operation,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        operation(np.random.default_rng(0))


@pytest.mark.parametrize(
    "operation",
    [
        lambda rng: drop_radar_frames(pd.DataFrame(), rate=1.01, rng=rng),
        lambda rng: drop_rf_bursts(pd.DataFrame(), rate="1.01", rng=rng),
    ],
)
def test_drop_helpers_reject_rates_above_one_before_empty_return(operation) -> None:
    with pytest.raises(ValueError, match="rate must not exceed 1"):
        operation(np.random.default_rng(0))
