from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_sequence_gate import blend_track5_estimate_sequence_gate


def test_estimate_sequence_gate_rejects_duplicate_sequence_weights() -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [2],
        }
    )
    base = pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [0.0],
            "state_x_m": [0.0],
            "state_y_m": [0.0],
            "state_z_m": [0.0],
        }
    )
    alternate = base.assign(state_x_m=10.0, state_y_m=10.0, state_z_m=10.0)
    duplicate_weights = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "weight": [0.0, 1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match=r"duplicate normalized sequence_id: seq0001",
    ):
        blend_track5_estimate_sequence_gate(
            base_estimates=base,
            alternate_estimates=alternate,
            template=template,
            sequence_weights=duplicate_weights,
        )
