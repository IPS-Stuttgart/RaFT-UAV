from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def _submission_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 100.0, 200.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    "iterations",
    [0, -1, 1.5, True, False, math.nan, math.inf, "not-an-integer"],
)
def test_speed_limit_rejects_invalid_iteration_counts(iterations: object) -> None:
    with pytest.raises(ValueError, match="iterations must be a positive integer"):
        project_track5_speed_limit(
            _submission_rows(),
            max_speed_mps=10.0,
            iterations=iterations,  # type: ignore[arg-type]
        )


def test_speed_limit_accepts_integer_equivalent_iteration_count() -> None:
    limited, diagnostics = project_track5_speed_limit(
        _submission_rows(),
        max_speed_mps=10.0,
        iterations=2.0,  # type: ignore[arg-type]
    )

    assert limited["state_x_m"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert diagnostics["output_speed_prev_mps"].dropna().max() <= 10.0
