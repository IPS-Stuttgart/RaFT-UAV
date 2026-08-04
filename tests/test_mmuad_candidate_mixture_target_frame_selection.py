from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    _target_time_candidate_groups,
    run_candidate_mixture_map,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "radar",
                "track_id": "past-a",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.1,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "past-b",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.2,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.2,
                "source": "radar",
                "track_id": "future-a",
                "x_m": 100.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.9,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.2,
                "source": "lidar_360",
                "track_id": "future-b",
                "x_m": 101.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.8,
                "predicted_sigma_m": 1.0,
            },
        ]
    )


def test_target_tolerance_selects_one_complete_nearest_frame() -> None:
    rows = _candidate_rows()
    groups = _target_time_candidate_groups(
        rows,
        candidate_times=rows["time_s"].to_numpy(float),
        target_times=np.asarray([0.05, 0.10, 0.16]),
        tolerance_s=0.2,
    )

    assert set(groups[0][1]["track_id"].astype(str)) == {"past-a", "past-b"}
    assert set(groups[1][1]["track_id"].astype(str)) == {"past-a", "past-b"}
    assert set(groups[2][1]["track_id"].astype(str)) == {"future-a", "future-b"}


def test_target_tolerance_does_not_apply_top_k_across_neighbor_frames() -> None:
    target_template = pd.DataFrame(
        {
            "Sequence": ["seqA"],
            "Timestamp": [0.05],
        }
    )

    result = run_candidate_mixture_map(
        _candidate_rows(),
        target_template=target_template,
        config=CandidateMixtureMapConfig(
            top_k=1,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            sigma_log_weight=0.0,
            smoothness_weight=0.0,
            target_time_tolerance_s=0.2,
            iterations=1,
        ),
    )

    assert result.assignments["track_id"].astype(str).tolist() == ["past-b"]
    assert result.estimates["state_x_m"].iloc[0] < 2.0
