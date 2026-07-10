from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


def test_anchor_blend_does_not_reintroduce_speed_violations() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 100.0, 200.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )

    limited, diagnostics = project_track5_speed_limit(
        submission,
        max_speed_mps=10.0,
        iterations=1,
        anchor_blend=0.5,
    )

    assert diagnostics["output_speed_prev_mps"].dropna().max() <= 10.0 + 1.0e-9
    assert limited["state_x_m"].tolist() == [0.0, 10.0, 20.0]
