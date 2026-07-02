import numpy as np
import pandas as pd
import pytest

from raft_uav.stress.perturbations import drop_radar_frames, drop_rf_bursts, scale_covariance_columns


def test_radar_frame_sampling_keeps_rows_without_frame_key():
    radar = pd.DataFrame(
        {
            "frame_index": [0.0, 1.0, np.nan],
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 1.0, 2.0],
            "north_m": [0.0, 0.0, 0.0],
            "up_m": [0.0, 0.0, 0.0],
        }
    )

    perturbed = drop_radar_frames(radar, rate=1.0, rng=np.random.default_rng(0))

    assert perturbed["frame_index"].isna().tolist() == [True]
    assert perturbed["time_s"].tolist() == [2.0]


def test_rf_burst_sampling_keeps_nonfinite_timestamps():
    rf = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, np.nan, float("inf")],
            "east_m": [0.0, 1.0, 2.0, 3.0],
            "north_m": [0.0, 0.0, 0.0, 0.0],
        }
    )

    perturbed = drop_rf_bursts(rf, rate=1.0, rng=np.random.default_rng(0))

    assert perturbed["east_m"].tolist() == [2.0, 3.0]


def test_rf_burst_sampling_all_nonfinite_timestamps_returns_copy():
    rf = pd.DataFrame(
        {
            "time_s": [np.nan, float("inf")],
            "east_m": [2.0, 3.0],
            "north_m": [0.0, 0.0],
        }
    )

    perturbed = drop_rf_bursts(rf, rate=1.0, rng=np.random.default_rng(0))

    pd.testing.assert_frame_equal(perturbed, rf)


def test_covariance_scaling_rejects_invalid_scales():
    frame = pd.DataFrame({"cov_xx": [1.0], "association_cov_yy": [2.0]})

    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and positive"):
            scale_covariance_columns(frame, scale=scale)


def test_covariance_scaling_only_scales_covariance_columns():
    frame = pd.DataFrame({"cov_xx": [1.0], "association_cov_yy": [2.0], "east_m": [3.0]})

    scaled = scale_covariance_columns(frame, scale=2.5)

    assert scaled["cov_xx"].tolist() == [2.5]
    assert scaled["association_cov_yy"].tolist() == [5.0]
    assert scaled["east_m"].tolist() == [3.0]
