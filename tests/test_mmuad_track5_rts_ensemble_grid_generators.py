from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd

import raft_uav.mmuad.track5_rts_ensemble_grid as grid_module
from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput


def test_rts_ensemble_grid_materializes_one_shot_parameter_grids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame({"placeholder": [1]}).to_csv(estimate_csv, index=False)

    seen: list[tuple[float, float, float]] = []

    def fake_build_track5_rts_ensemble(
        loaded_inputs,
        template,
        *,
        measurement_sigma_m: float,
        process_accel_std_mps2: float,
        initial_position_std_m: float,
        initial_velocity_std_mps: float,
        spread_variance_scale: float,
        max_nearest_time_delta_s: float | None,
    ):
        del loaded_inputs
        del template
        del initial_position_std_m
        del initial_velocity_std_mps
        del max_nearest_time_delta_s
        seen.append(
            (
                float(measurement_sigma_m),
                float(process_accel_std_mps2),
                float(spread_variance_scale),
            )
        )
        estimates = pd.DataFrame(
            {
                "sequence_id": ["seq0001"],
                "time_s": [0.0],
                "state_x_m": [0.0],
                "state_y_m": [0.0],
                "state_z_m": [0.0],
            }
        )
        diagnostics = pd.DataFrame(
            {
                "valid_input_count": [1],
                "input_spread_m": [0.0],
            }
        )
        return estimates, diagnostics

    monkeypatch.setattr(
        grid_module,
        "build_track5_rts_ensemble",
        fake_build_track5_rts_ensemble,
    )

    measurement_sigmas = (value for value in (5.0, 10.0))
    process_accels = (value for value in (1.0, 3.0))
    spread_scales = (value for value in (0.0, 2.0))
    grid, best = grid_module.run_track5_rts_ensemble_grid_search(
        [EstimateInput(label="base", path=estimate_csv)],
        template=pd.DataFrame(),
        truth=pd.DataFrame(
            {
                "sequence_id": ["seq0001"],
                "time_s": [0.0],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [0.0],
            }
        ),
        measurement_sigma_grid=measurement_sigmas,
        process_accel_grid=process_accels,
        spread_variance_scale_grid=spread_scales,
    )

    expected = set(product((5.0, 10.0), (1.0, 3.0), (0.0, 2.0)))
    assert len(grid) == len(expected)
    assert set(seen) == expected
    assert best["best"]["matched_row_count"] == 1
