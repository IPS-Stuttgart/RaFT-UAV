from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.leaderboard import build_mmuad_leaderboard
from raft_uav.mmuad.leaderboard import rank_leaderboard_frame


def test_leaderboard_ranks_valid_score_before_blocked_subset_score() -> None:
    frame = pd.DataFrame(
        {
            "method": ["blocked-partial", "valid"],
            "score_valid_for_leaderboard": [False, True],
            "leaderboard_ready": [False, True],
            "pose_mse_loss_m2": [0.0, 4.0],
            "rmse_3d_m": [0.0, 2.0],
            "p95_3d_m": [0.0, 3.0],
        }
    )

    ranked = rank_leaderboard_frame(frame)

    assert ranked["method"].tolist() == ["valid", "blocked-partial"]
    assert ranked["rank"].tolist() == [1, 2]
    assert ranked["rank_metric"].tolist() == ["pose_mse_loss_m2"] * 2


def test_blocked_score_does_not_force_metric_for_valid_rows() -> None:
    frame = pd.DataFrame(
        {
            "method": ["blocked", "valid-worse", "valid-better"],
            "score_valid_for_leaderboard": [False, True, True],
            "pose_mse_loss_m2": [0.0, np.nan, np.nan],
            "rmse_3d_m": [0.1, 5.0, 2.0],
            "p95_3d_m": [0.1, 6.0, 3.0],
        }
    )

    ranked = rank_leaderboard_frame(frame)

    assert ranked["method"].tolist() == [
        "valid-better",
        "valid-worse",
        "blocked",
    ]
    assert ranked["rank_metric"].tolist() == ["rmse_3d_m"] * 3


def test_missing_new_validity_flag_falls_back_to_legacy_metadata() -> None:
    frame = pd.DataFrame(
        {
            "method": ["blocked-partial", "legacy-valid"],
            "score_valid_for_leaderboard": [False, np.nan],
            "leaderboard_ready": [False, True],
            "pose_mse_loss_m2": [0.0, 4.0],
            "rmse_3d_m": [0.0, 2.0],
            "p95_3d_m": [0.0, 3.0],
        }
    )

    ranked = rank_leaderboard_frame(frame)

    assert ranked["method"].tolist() == ["legacy-valid", "blocked-partial"]
    assert ranked["rank"].tolist() == [1, 2]


def test_public_leaderboard_builder_uses_validity_aware_ranker() -> None:
    assert (
        build_mmuad_leaderboard.__globals__["rank_leaderboard_frame"]
        is rank_leaderboard_frame
    )
