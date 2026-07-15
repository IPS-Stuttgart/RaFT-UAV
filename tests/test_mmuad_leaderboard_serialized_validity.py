from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.leaderboard import rank_leaderboard_frame


def test_leaderboard_accepts_serialized_numeric_true_flag() -> None:
    frame = pd.DataFrame(
        {
            "method": ["blocked", "valid"],
            "score_valid_for_leaderboard": ["0.0", "1.0"],
            "pose_mse_loss_m2": [0.0, 4.0],
            "p95_3d_m": [0.0, 5.0],
        }
    )

    ranked = rank_leaderboard_frame(frame)

    assert ranked["method"].tolist() == ["valid", "blocked"]
    assert ranked["rank"].tolist() == [1, 2]
