from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.radar_calibration_audit import (
    IDENTITY_CALIBRATION,
    apply_spatial_calibration,
    evaluate_calibrated_measurements,
    fit_constant_offset,
    fit_time_offset,
    fit_yaw_offset_altitude,
    pair_measurements_to_truth,
)


def test_constant_offset_model_removes_translation_bias() -> None:
    truth = _truth_frame()
    measurements = truth.copy()
    measurements["east_m"] += 12.0
    measurements["north_m"] -= 8.0
    measurements["up_m"] += 5.0

    pairs = pair_measurements_to_truth(measurements, truth)
    model = fit_constant_offset(pairs)
    corrected = apply_spatial_calibration(measurements, model)

    np.testing.assert_allclose(corrected[["east_m", "north_m", "up_m"]], truth[["east_m", "north_m", "up_m"]])
    assert model.offset_east_m == -12.0
    assert model.offset_north_m == 8.0
    assert model.offset_up_m == -5.0


def test_yaw_offset_altitude_model_corrects_horizontal_rotation() -> None:
    truth = _truth_frame()
    yaw_rad = np.deg2rad(7.0)
    c = np.cos(-yaw_rad)
    s = np.sin(-yaw_rad)
    inverse_rotation = np.array([[c, s], [-s, c]])
    translation = np.array([20.0, -15.0])
    measured_xy = (truth[["east_m", "north_m"]].to_numpy(dtype=float) - translation) @ inverse_rotation.T
    measurements = truth.copy()
    measurements["east_m"] = measured_xy[:, 0]
    measurements["north_m"] = measured_xy[:, 1]
    measurements["up_m"] = truth["up_m"] - 4.0

    model = fit_yaw_offset_altitude(pair_measurements_to_truth(measurements, truth))
    metrics = evaluate_calibrated_measurements(measurements, truth, calibration=model)

    assert abs(np.rad2deg(model.yaw_rad) + 7.0) < 1.0e-9
    assert metrics["rmse_m"] < 1.0e-9
    assert model.offset_up_m == 4.0


def test_time_offset_fit_recovers_training_shift() -> None:
    truth = _truth_frame()
    measurements = truth.copy()
    measurements["time_s"] = measurements["time_s"] - 1.0
    best, sweep = fit_time_offset(
        {"Opt1": measurements},
        {"Opt1": truth},
        offsets_s=[-1.0, 0.0, 1.0],
        calibration=IDENTITY_CALIBRATION,
    )

    assert best == 1.0
    assert set(sweep["time_offset_s"]) == {-1.0, 0.0, 1.0}


def test_time_offset_fit_skips_candidates_without_matches() -> None:
    truth = _truth_frame()
    measurements = truth.copy()
    measurements["time_s"] = measurements["time_s"] - 1.0

    best, sweep = fit_time_offset(
        {"Opt1": measurements},
        {"Opt1": truth},
        offsets_s=[100.0, 1.0],
        calibration=IDENTITY_CALIBRATION,
        max_time_delta_s=0.5,
    )

    assert best == 1.0
    invalid = sweep.loc[sweep["time_offset_s"] == 100.0].iloc[0]
    assert invalid["matched_rows"] == 0.0
    assert np.isnan(invalid["rmse_m"])


def test_time_offset_fit_raises_when_all_candidates_have_no_matches() -> None:
    truth = _truth_frame()
    measurements = truth.copy()
    measurements["time_s"] = measurements["time_s"] + 100.0

    with pytest.raises(RuntimeError, match="no finite time-offset candidates"):
        fit_time_offset(
            {"Opt1": measurements},
            {"Opt1": truth},
            offsets_s=[0.0, 1.0],
            calibration=IDENTITY_CALIBRATION,
            max_time_delta_s=0.5,
        )


def _truth_frame() -> pd.DataFrame:
    time_s = np.arange(0.0, 8.0)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "east_m": 3.0 * time_s + 5.0,
            "north_m": 0.5 * time_s**2 - 2.0,
            "up_m": 100.0 + 0.25 * time_s,
        }
    )
