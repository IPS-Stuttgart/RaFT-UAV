from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_rts_ensemble import build_track5_rts_ensemble


def _estimate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 3,
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 100.0, 0.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
        }
    )


def _template(times: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * len(times),
            "Timestamp": times,
        }
    )


def test_rts_duplicate_template_rows_do_not_reweight_measurements() -> None:
    kwargs = {
        "measurement_sigma_m": 10.0,
        "process_accel_std_mps2": 1.0,
        "initial_position_std_m": 10.0,
        "initial_velocity_std_mps": 10.0,
        "spread_variance_scale": 0.0,
        "max_nearest_time_delta_s": 0.0,
    }
    estimate_inputs = [("estimate", _estimate_rows(), 1.0)]

    baseline, baseline_diagnostics = build_track5_rts_ensemble(
        estimate_inputs,
        _template([0.0, 1.0, 2.0]),
        **kwargs,
    )
    duplicated, duplicated_diagnostics = build_track5_rts_ensemble(
        estimate_inputs,
        _template([0.0, 1.0, 1.0, 2.0]),
        **kwargs,
    )

    coordinate_columns = ["state_x_m", "state_y_m", "state_z_m"]
    duplicated_unique = duplicated.drop_duplicates(
        ["sequence_id", "time_s"]
    )
    np.testing.assert_allclose(
        duplicated_unique[coordinate_columns].to_numpy(float),
        baseline[coordinate_columns].to_numpy(float),
        rtol=0.0,
        atol=1.0e-12,
    )

    duplicated_middle = duplicated.loc[
        duplicated["time_s"] == 1.0,
        coordinate_columns,
    ].to_numpy(float)
    np.testing.assert_allclose(
        duplicated_middle,
        np.repeat(
            baseline.loc[baseline["time_s"] == 1.0, coordinate_columns]
            .to_numpy(float),
            2,
            axis=0,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )

    diagnostic_numeric_columns = [
        "valid_input_count",
        "weighted_x_m",
        "weighted_y_m",
        "weighted_z_m",
        "inverse_variance_weight_sum",
        "measurement_variance_m2",
        "input_spread_m",
        "smoothed_x_m",
        "smoothed_y_m",
        "smoothed_z_m",
        "smoothed_minus_weighted_m",
    ]
    duplicated_diagnostics_unique = duplicated_diagnostics.drop_duplicates(
        ["sequence_id", "time_s"]
    )
    np.testing.assert_allclose(
        duplicated_diagnostics_unique[diagnostic_numeric_columns].to_numpy(float),
        baseline_diagnostics[diagnostic_numeric_columns].to_numpy(float),
        rtol=0.0,
        atol=1.0e-12,
        equal_nan=True,
    )
    assert duplicated_diagnostics_unique["input_labels"].tolist() == (
        baseline_diagnostics["input_labels"].tolist()
    )

    duplicate_middle_diagnostics = duplicated_diagnostics.loc[
        duplicated_diagnostics["time_s"] == 1.0
    ]
    assert duplicate_middle_diagnostics["valid_input_count"].tolist() == [1, 1]
    assert duplicate_middle_diagnostics["input_labels"].tolist() == [
        "estimate",
        "estimate",
    ]
    np.testing.assert_allclose(
        duplicate_middle_diagnostics["measurement_variance_m2"].to_numpy(float),
        np.repeat(
            baseline_diagnostics.loc[
                baseline_diagnostics["time_s"] == 1.0,
                "measurement_variance_m2",
            ].to_numpy(float),
            2,
        ),
        rtol=0.0,
        atol=1.0e-12,
    )
    assert len(duplicated) == 4
    assert len(duplicated_diagnostics) == 4
