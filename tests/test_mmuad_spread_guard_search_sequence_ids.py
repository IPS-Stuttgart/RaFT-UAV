from pathlib import Path

import pandas as pd

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_spread_guard_search import search_track5_spread_guard_settings


def test_spread_guard_search_preserves_padded_estimate_sequence_ids(tmp_path: Path) -> None:
    estimate_csv = tmp_path / "estimate.csv"
    pd.DataFrame(
        {
            "Sequence": ["001"],
            "Timestamp": [10.0],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
        }
    ).to_csv(estimate_csv, index=False)

    template = pd.DataFrame({"Sequence": ["001"], "Timestamp": [10.0]})
    truth = pd.DataFrame(
        {
            "sequence_id": ["001"],
            "time_s": [10.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    grid, best = search_track5_spread_guard_settings(
        [EstimateInput(label="candidate", path=estimate_csv, weight=1.0)],
        template=template,
        truth=truth,
        spread_thresholds_m=[0.0],
    )

    assert grid.loc[0, "matched_rows"] == 1
    assert grid.loc[0, "pose_mse_m2"] == 0.0
    assert best["metrics"]["matched_rows"] == 1
