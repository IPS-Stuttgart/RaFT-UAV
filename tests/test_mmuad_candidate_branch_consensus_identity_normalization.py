from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_branch_consensus import (
    attach_candidate_branch_consensus,
)
from raft_uav.mmuad.schema import CandidateFrame


def test_equivalent_source_and_branch_labels_do_not_create_cross_sensor_support() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": [" RF ", "rf"],
            "track_id": ["higher", "lower"],
            "candidate_branch": [" RAW ", "raw"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "ranker_score": [0.9, 0.1],
        }
    )

    augmented = attach_candidate_branch_consensus(
        CandidateFrame(rows),
        time_window_s=0.1,
        distance_gate_m=5.0,
    ).rows.set_index("track_id")

    assert augmented["branch_consensus_neighbor_count"].tolist() == [0, 0]
    assert augmented[
        "branch_consensus_nearest_cross_source_distance_m"
    ].isna().all()
    assert augmented.loc[
        "higher", "branch_consensus_base_score_normalized"
    ] == pytest.approx(1.0)
    assert augmented.loc[
        "lower", "branch_consensus_base_score_normalized"
    ] == pytest.approx(0.0)
    assert augmented.loc["higher", "source"] == "RF"
    assert augmented.loc["lower", "source"] == "rf"
    assert augmented.loc["higher", "candidate_branch"] == "RAW"
    assert augmented.loc["lower", "candidate_branch"] == "raw"


def test_pair_advantage_groups_equivalent_source_labels() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq"],
            "time_s": [0.0, 0.0, 0.0],
            "source": [" RF ", "rf", "lidar"],
            "track_id": ["raw", "calibrated", "lidar"],
            "candidate_branch": [" RAW ", "source_translation", "raw"],
            "mmuad_calibration_origin_row": [7, 7, 12],
            "x_m": [10.0, 1.0, 1.2],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [0.9, 0.4, 0.5],
        }
    )

    augmented = attach_candidate_branch_consensus(
        CandidateFrame(rows),
        time_window_s=0.01,
        distance_gate_m=2.0,
        distance_scale_m=2.0,
    ).rows.set_index("track_id")

    assert augmented.loc[
        "raw", "branch_consensus_nearest_cross_source_distance_m"
    ] == pytest.approx(8.8)
    assert augmented.loc[
        "calibrated", "branch_consensus_nearest_cross_source_distance_m"
    ] == pytest.approx(0.2)
    assert augmented.loc["raw", "branch_consensus_pair_advantage_m"] < 0.0
    assert augmented.loc["calibrated", "branch_consensus_pair_advantage_m"] > 0.0
    assert (
        augmented.loc[
            "calibrated",
            "branch_consensus_nearest_cross_source",
        ]
        == "lidar"
    )
