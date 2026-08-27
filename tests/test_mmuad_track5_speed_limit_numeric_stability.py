from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import _clip_to_speed_ball
from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _two_point_submission(
    *,
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 1.0],
            "state_x_m": [start_xyz[0], end_xyz[0]],
            "state_y_m": [start_xyz[1], end_xyz[1]],
            "state_z_m": [start_xyz[2], end_xyz[2]],
            "Classification": [1, 1],
        }
    )


def test_speed_limit_projects_large_diagonal_jump_to_boundary() -> None:
    submission = _two_point_submission(
        start_xyz=(0.0, 0.0, 0.0),
        end_xyz=(1.0e308, 1.0e308, 0.0),
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        limited, diagnostics = project_track5_speed_limit(
            submission,
            max_speed_mps=1.0e308,
            iterations=1,
        )

    expected_component = 1.0e308 / np.sqrt(2.0)
    projected = limited.loc[1, ["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    assert projected == pytest.approx(
        [expected_component, expected_component, 0.0],
        rel=2.0e-15,
    )
    assert diagnostics.loc[1, "input_speed_prev_mps"] == pytest.approx(
        np.sqrt(2.0) * 1.0e308,
        rel=2.0e-15,
    )
    assert diagnostics.loc[1, "output_speed_prev_mps"] <= 1.0e308 * (1.0 + 2.0e-15)


def test_speed_limit_handles_opposite_large_coordinates_without_nan() -> None:
    submission = _two_point_submission(
        start_xyz=(-1.0e308, 0.0, 0.0),
        end_xyz=(1.0e308, 0.0, 0.0),
    )

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        limited, diagnostics = project_track5_speed_limit(
            submission,
            max_speed_mps=1.0e308,
            iterations=1,
        )

    coordinates = limited[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    assert np.isfinite(coordinates).all()
    assert limited.loc[1, "state_x_m"] == 0.0
    assert diagnostics.loc[1, "input_speed_prev_mps"] == np.finfo(float).max
    assert diagnostics.loc[1, "output_speed_prev_mps"] == 1.0e308
    assert diagnostics.loc[1, "speed_limit_correction_m"] == 1.0e308


def test_speed_limit_preserves_ordinary_projection_exactly() -> None:
    projected = _clip_to_speed_ball(
        np.array([6.0, 8.0, 0.0]),
        np.zeros(3),
        1.0,
        5.0,
    )

    assert np.array_equal(projected, np.array([3.0, 4.0, 0.0]))
