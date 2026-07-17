from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_sequence_gate import blend_track5_sequence_gate


def _submission(x_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [x_m],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
            "Classification": [1],
        }
    )


def test_sequence_gate_rejects_invalid_duplicate_weights_before_averaging() -> None:
    weights = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "weight": [-1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match=r"sequence weight for seq0001.*\[0, 1\]"):
        blend_track5_sequence_gate(
            base_submission=_submission(0.0),
            alternate_submission=_submission(10.0),
            sequence_weights=weights,
        )


def test_sequence_gate_keeps_averaging_valid_duplicate_weights() -> None:
    result = blend_track5_sequence_gate(
        base_submission=_submission(0.0),
        alternate_submission=_submission(10.0),
        sequence_weights=pd.DataFrame(
            {
                "sequence_id": ["seq0001", "seq0001"],
                "weight": [0.25, 0.75],
            }
        ),
    )

    assert result.estimates.loc[0, "sequence_gate_weight"] == pytest.approx(0.5)
    assert result.estimates.loc[0, "state_x_m"] == pytest.approx(5.0)


@pytest.mark.parametrize("value", [True, np.bool_(False), np.array([0.5]), 0.5 + 0.0j])
def test_sequence_gate_rejects_non_real_scalar_default_weights(value: object) -> None:
    with pytest.raises(ValueError, match=r"default_weight.*\[0, 1\]"):
        blend_track5_sequence_gate(
            base_submission=_submission(0.0),
            alternate_submission=_submission(10.0),
            sequence_weights=pd.DataFrame(),
            default_weight=value,
        )
