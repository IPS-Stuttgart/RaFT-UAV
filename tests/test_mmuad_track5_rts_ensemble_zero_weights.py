from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_rts_ensemble import build_track5_rts_ensemble


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001"] * 3,
            "Timestamp": [0.0, 1.0, 2.0],
            "Position": ["(0,0,0)"] * 3,
            "Classification": [2] * 3,
        }
    )


def _estimate(offset: float) -> pd.DataFrame:
    times = np.arange(3, dtype=float)
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 3,
            "time_s": times,
            "state_x_m": times + offset,
            "state_y_m": np.zeros(3),
            "state_z_m": np.ones(3),
        }
    )


def test_rts_ensemble_ignores_zero_weight_input() -> None:
    expected, _ = build_track5_rts_ensemble(
        [("active", _estimate(0.0), 1.0)],
        _template(),
        measurement_sigma_m=1.0,
        process_accel_std_mps2=0.1,
    )
    actual, diagnostics = build_track5_rts_ensemble(
        [
            ("disabled", _estimate(1000.0), 0.0),
            ("active", _estimate(0.0), 1.0),
        ],
        _template(),
        measurement_sigma_m=1.0,
        process_accel_std_mps2=0.1,
    )

    np.testing.assert_allclose(
        actual[["state_x_m", "state_y_m", "state_z_m"]],
        expected[["state_x_m", "state_y_m", "state_z_m"]],
    )
    assert diagnostics["valid_input_count"].tolist() == [1, 1, 1]
    assert diagnostics["input_labels"].tolist() == ["active", "active", "active"]


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_rts_ensemble_still_rejects_invalid_weights(weight: float) -> None:
    with pytest.raises(ValueError, match="non-negative and finite"):
        build_track5_rts_ensemble(
            [("bad", _estimate(0.0), weight)],
            _template(),
        )
