from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_vertical_repair import repair_track5_vertical_spikes


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 100.0, 2.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize(
    "parameter",
    [
        "max_vertical_speed_mps",
        "max_neighbor_vertical_speed_mps",
        "max_vertical_residual_m",
        "max_horizontal_speed_mps",
    ],
)
@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf, True, False])
def test_vertical_repair_rejects_invalid_thresholds(
    parameter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=rf"{parameter} must be finite and non-negative"):
        repair_track5_vertical_spikes(_submission(), **{parameter: value})


def test_vertical_repair_accepts_zero_thresholds_and_disabled_horizontal_gate() -> None:
    repaired, diagnostics = repair_track5_vertical_spikes(
        _submission(),
        max_vertical_speed_mps=0.0,
        max_neighbor_vertical_speed_mps=0.0,
        max_vertical_residual_m=0.0,
        max_horizontal_speed_mps=None,
    )

    assert len(repaired) == 3
    assert not diagnostics.empty
