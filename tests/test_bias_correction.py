from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.calibration.bias import (
    RADAR_TARGET_COLUMNS,
    RF_TARGET_COLUMNS,
    apply_bias_correction_models,
    bias_correction_summary,
    fit_sensor_bias_correction,
    load_bias_correction_models,
    save_bias_correction_models,
)


def _truth() -> pd.DataFrame:
    time_s = np.arange(8, dtype=float)
    return pd.DataFrame(
        {
            "time_s": time_s,
            "east_m": 10.0 + 4.0 * time_s,
            "north_m": -5.0 + 2.0 * time_s,
            "up_m": 100.0 + time_s,
        }
    )


def test_rf_bias_model_learns_and_applies_linear_residual() -> None:
    truth = _truth()
    time_s = truth["time_s"].to_numpy(dtype=float)
    rf = pd.DataFrame(
        {
            "time_s": time_s,
            "east_m": truth["east_m"].to_numpy(dtype=float) + 5.0 + 0.5 * time_s,
            "north_m": truth["north_m"].to_numpy(dtype=float) - 3.0,
            "std_m": np.full_like(time_s, 20.0),
        }
    )

    model = fit_sensor_bias_correction(
        rf,
        truth,
        source="rf",
        target_columns=RF_TARGET_COLUMNS,
        feature_columns=("time_s",),
        time_gate_s=0.25,
        ridge_alpha=0.0,
        min_samples=3,
    )
    corrected = model.apply(rf)

    np.testing.assert_allclose(corrected["east_m"], truth["east_m"], atol=1e-9)
    np.testing.assert_allclose(corrected["north_m"], truth["north_m"], atol=1e-9)
    assert "raw_east_m" in corrected.columns
    assert "bias_east_m" in corrected.columns


def test_radar_bias_model_corrects_3d_positions() -> None:
    truth = _truth()
    time_s = truth["time_s"].to_numpy(dtype=float)
    radar = pd.DataFrame(
        {
            "time_s": time_s,
            "east_m": truth["east_m"].to_numpy(dtype=float) + 7.0,
            "north_m": truth["north_m"].to_numpy(dtype=float) - 4.0,
            "up_m": truth["up_m"].to_numpy(dtype=float) + 11.0,
            "range_m": 100.0 + time_s,
            "cat_prob_uav": np.ones_like(time_s),
        }
    )

    model = fit_sensor_bias_correction(
        radar,
        truth,
        source="radar",
        target_columns=RADAR_TARGET_COLUMNS,
        feature_columns=("time_s",),
        time_gate_s=0.25,
        ridge_alpha=0.0,
        min_samples=3,
    )
    corrected = model.apply(radar)

    np.testing.assert_allclose(
        corrected[["east_m", "north_m", "up_m"]],
        truth[["east_m", "north_m", "up_m"]],
        atol=1e-9,
    )


def test_bias_model_bundle_round_trips(tmp_path: Path) -> None:
    truth = _truth()
    rf = pd.DataFrame(
        {
            "time_s": truth["time_s"],
            "east_m": truth["east_m"] + 2.0,
            "north_m": truth["north_m"] - 1.0,
            "std_m": 30.0,
        }
    )
    radar = pd.DataFrame(
        {
            "time_s": truth["time_s"],
            "east_m": truth["east_m"] + 3.0,
            "north_m": truth["north_m"] - 2.0,
            "up_m": truth["up_m"] + 5.0,
        }
    )

    rf_model = fit_sensor_bias_correction(
        rf,
        truth,
        source="rf",
        target_columns=RF_TARGET_COLUMNS,
        feature_columns=("time_s",),
        ridge_alpha=0.0,
    )
    radar_model = fit_sensor_bias_correction(
        radar,
        truth,
        source="radar",
        target_columns=RADAR_TARGET_COLUMNS,
        feature_columns=("time_s",),
        ridge_alpha=0.0,
    )
    path = tmp_path / "bias.json"
    save_bias_correction_models({"rf": rf_model, "radar": radar_model}, path)
    loaded = load_bias_correction_models(path)
    corrected_rf, corrected_radar = apply_bias_correction_models(rf=rf, radar=radar, models=loaded)

    np.testing.assert_allclose(
        corrected_rf[["east_m", "north_m"]], truth[["east_m", "north_m"]], atol=1e-9
    )
    np.testing.assert_allclose(
        corrected_radar[["east_m", "north_m", "up_m"]],
        truth[["east_m", "north_m", "up_m"]],
        atol=1e-9,
    )
    summary = bias_correction_summary(loaded)
    assert summary["rf"]["training_rows"] == len(rf)
    assert summary["radar"]["training_rows"] == len(radar)
