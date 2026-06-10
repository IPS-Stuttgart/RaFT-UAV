import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import AsyncConstantVelocityKalmanTracker
from raft_uav.baselines.radar_covariance import (
    BIAS_RESIDUAL_INCLUDED_COLUMN,
    RadarCovarianceConfig,
    append_radar_covariance_columns,
    row_radar_covariance,
)
from raft_uav.baselines.tracklet_viterbi_range_covariance import _radar_row_covariance
from raft_uav.baselines.radar_association import _nis_scored_candidates, _row_covariance
from raft_uav.calibration.bias import BIAS_RESIDUAL_STD_COLUMN_PREFIX
from raft_uav.calibration.empirical_covariance import aligned_residuals


def test_range_angle_radar_covariance_grows_with_range():
    radar = pd.DataFrame(
        [
            {"east_m": 100.0, "north_m": 0.0, "up_m": 0.0, "range_m": 100.0},
            {"east_m": 1000.0, "north_m": 0.0, "up_m": 0.0, "range_m": 1000.0},
        ]
    )
    annotated = append_radar_covariance_columns(
        radar,
        RadarCovarianceConfig(
            mode="range-angle",
            range_std_m=1.0,
            azimuth_std_deg=2.0,
            elevation_std_deg=2.0,
            min_std_m=1.0,
        ),
    )

    near_covariance = row_radar_covariance(annotated.iloc[0])
    far_covariance = row_radar_covariance(annotated.iloc[1])

    assert near_covariance is not None
    assert far_covariance is not None
    assert far_covariance[1, 1] > near_covariance[1, 1] * 50.0
    assert annotated["association_covariance_mode"].tolist() == ["range-angle", "range-angle"]


def test_nis_scoring_uses_candidate_specific_covariance_columns():
    tracker = AsyncConstantVelocityKalmanTracker(initial_position=np.zeros(3), initial_time_s=0.0)
    tracker.covariance = np.zeros((6, 6))
    candidates = pd.DataFrame(
        [
            {
                "track_id": 1,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_cov_ee": 1.0,
                "association_cov_nn": 1.0,
                "association_cov_uu": 1.0,
                "association_cov_en": 0.0,
                "association_cov_eu": 0.0,
                "association_cov_nu": 0.0,
            },
            {
                "track_id": 2,
                "time_s": 0.0,
                "east_m": 10.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "association_cov_ee": 100.0,
                "association_cov_nn": 1.0,
                "association_cov_uu": 1.0,
                "association_cov_en": 0.0,
                "association_cov_eu": 0.0,
                "association_cov_nu": 0.0,
            },
        ]
    )

    scored = _nis_scored_candidates(candidates, tracker, np.eye(3))

    assert scored.loc[scored["track_id"] == 1, "association_nis"].iloc[0] == 100.0
    assert scored.loc[scored["track_id"] == 2, "association_nis"].iloc[0] == 1.0


def test_radar_association_rejects_indefinite_row_covariance():
    row = pd.Series(
        {
            "association_cov_ee": 1.0,
            "association_cov_nn": 1.0,
            "association_cov_uu": 1.0,
            "association_cov_en": 2.0,
            "association_cov_eu": 0.0,
            "association_cov_nu": 0.0,
        }
    )

    assert _row_covariance(row) is None


def test_association_row_covariance_rejects_indefinite_primary_covariance():
    row = pd.Series(
        {
            "association_cov_ee": 1.0,
            "association_cov_nn": 1.0,
            "association_cov_uu": 1.0,
            "association_cov_en": 2.0,
            "association_cov_eu": 0.0,
            "association_cov_nu": 0.0,
            "cov_ee": 4.0,
            "cov_nn": 5.0,
            "cov_uu": 6.0,
            "cov_en": 0.0,
            "cov_eu": 0.0,
            "cov_nu": 0.0,
        }
    )

    covariance = _row_covariance(row)

    assert covariance is not None
    np.testing.assert_allclose(covariance, np.diag([4.0, 5.0, 6.0]))


def test_tracklet_range_covariance_prefers_learned_row_covariance_columns():
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    row = pd.Series(
        {
            "range_m": 10_000.0,
            "cov_ee": 9.0,
            "cov_nn": 16.0,
            "cov_uu": 25.0,
            "cov_en": 1.0,
            "cov_eu": 2.0,
            "cov_nu": 3.0,
        }
    )

    covariance = _radar_row_covariance(row, default_covariance, object())

    np.testing.assert_allclose(
        covariance,
        [[9.0, 1.0, 2.0], [1.0, 16.0, 3.0], [2.0, 3.0, 25.0]],
    )


def test_bias_residual_std_inflates_fallback_radar_covariance():
    fallback = np.diag([25.0**2, 25.0**2, 35.0**2])
    row = pd.Series(
        {
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}east_m": 3.0,
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}north_m": 4.0,
            f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}up_m": 5.0,
        }
    )

    covariance = row_radar_covariance(row, fallback)

    assert covariance is not None
    np.testing.assert_allclose(np.diag(covariance), np.diag(fallback) + np.array([9.0, 16.0, 25.0]))


def test_bias_residual_std_is_added_once_to_range_angle_covariance():
    radar = pd.DataFrame(
        [
            {
                "east_m": 100.0,
                "north_m": 0.0,
                "up_m": 0.0,
                "range_m": 100.0,
                f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}east_m": 3.0,
                f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}north_m": 4.0,
                f"{BIAS_RESIDUAL_STD_COLUMN_PREFIX}up_m": 5.0,
            }
        ]
    )
    baseline = append_radar_covariance_columns(
        radar.drop(columns=[column for column in radar.columns if column.startswith(BIAS_RESIDUAL_STD_COLUMN_PREFIX)]),
        RadarCovarianceConfig(mode="range-angle", min_std_m=1.0),
    )
    annotated = append_radar_covariance_columns(
        radar,
        RadarCovarianceConfig(mode="range-angle", min_std_m=1.0),
    )

    baseline_covariance = row_radar_covariance(baseline.iloc[0])
    annotated_covariance = row_radar_covariance(annotated.iloc[0])

    assert baseline_covariance is not None
    assert annotated_covariance is not None
    assert bool(annotated.iloc[0][BIAS_RESIDUAL_INCLUDED_COLUMN])
    np.testing.assert_allclose(
        np.diag(annotated_covariance),
        np.diag(baseline_covariance) + np.array([9.0, 16.0, 25.0]),
        rtol=1e-6,
        atol=1e-6,
    )


def test_empirical_covariance_residual_alignment_handles_unsorted_truth_times():
    frame = pd.DataFrame(
        {
            "time_s": [0.05, 0.95],
            "east_m": [13.0, 18.0],
            "north_m": [4.0, 4.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [1.0, 0.0],
            "east_m": [20.0, 10.0],
            "north_m": [5.0, 3.0],
        }
    )

    residuals = aligned_residuals(
        frame,
        truth,
        source="rf",
        max_time_delta_s=0.1,
    )

    np.testing.assert_allclose(residuals, [[3.0, 1.0], [-2.0, -1.0]])
