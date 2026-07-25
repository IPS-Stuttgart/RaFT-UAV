from __future__ import annotations

import numpy as np
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
    ("field", "value", "message"),
    [
        (
            "max_speed_mps",
            np.asarray([10.0]),
            "max_speed_mps must be positive and finite",
        ),
        (
            "iterations",
            np.asarray([2]),
            "iterations must be a positive integer",
        ),
        (
            "anchor_blend",
            np.asarray([0.25]),
            r"anchor_blend must be finite and in \[0, 1\)",
        ),
    ],
)
def test_speed_limit_rejects_non_scalar_array_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "max_speed_mps": 10.0,
        "iterations": 2,
        "anchor_blend": 0.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        project_track5_speed_limit(_submission_rows(), **kwargs)


def test_speed_limit_accepts_zero_dimensional_numpy_controls() -> None:
    limited, diagnostics = project_track5_speed_limit(
        _submission_rows(),
        max_speed_mps=np.asarray(10.0),
        iterations=np.asarray(2),
        anchor_blend=np.asarray(0.0),
    )

    assert limited["state_x_m"].tolist() == pytest.approx([0.0, 10.0, 20.0])
    assert diagnostics["output_speed_prev_mps"].dropna().max() <= 10.0
