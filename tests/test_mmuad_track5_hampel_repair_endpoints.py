from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.track5_hampel_repair import repair_track5_hampel_spikes


def test_hampel_repair_preserves_sequence_endpoints() -> None:
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 5,
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0],
            "state_x_m": [100.0, 1.0, 2.0, 3.0, 100.0],
            "state_y_m": [0.0] * 5,
            "state_z_m": [5.0] * 5,
            "Classification": [2] * 5,
        }
    )

    repaired, diagnostics = repair_track5_hampel_spikes(
        submission,
        window_radius=2,
        sigma_threshold=2.0,
        min_scale_m=1.0,
        min_residual_m=5.0,
        repair_blend=1.0,
    )

    assert repaired["state_x_m"].tolist() == [100.0, 1.0, 2.0, 3.0, 100.0]
    assert not repaired["hampel_repair_applied"].any()

    endpoint_rows = diagnostics.loc[diagnostics["time_s"].isin([0.0, 4.0])]
    assert len(endpoint_rows) == 2
    assert not endpoint_rows["hampel_iteration_applied"].any()
    assert not endpoint_rows["hampel_repair_applied"].any()
