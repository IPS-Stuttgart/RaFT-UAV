from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.leaderboard import rank_leaderboard_frame


@pytest.mark.parametrize(
    "rank_metric",
    [
        "truth_coverage_fraction",
        "uav_type_accuracy",
        "classification_accuracy",
    ],
)
def test_leaderboard_ranks_higher_is_better_metrics_descending(rank_metric: str) -> None:
    frame = pd.DataFrame(
        {
            "method": ["worse", "better"],
            rank_metric: [0.25, 0.75],
            "p95_3d_m": [1.0, 1.0],
            "max_3d_m": [2.0, 2.0],
        }
    )

    ranked = rank_leaderboard_frame(frame, rank_metric=rank_metric)

    assert ranked["method"].tolist() == ["better", "worse"]
    assert ranked["rank_metric"].tolist() == [rank_metric, rank_metric]


def test_leaderboard_keeps_error_metrics_ascending() -> None:
    frame = pd.DataFrame(
        {
            "method": ["worse", "better"],
            "pose_mse_loss_m2": [4.0, 1.0],
            "p95_3d_m": [3.0, 2.0],
            "max_3d_m": [5.0, 4.0],
        }
    )

    ranked = rank_leaderboard_frame(frame, rank_metric="pose_mse_loss_m2")

    assert ranked["method"].tolist() == ["better", "worse"]
