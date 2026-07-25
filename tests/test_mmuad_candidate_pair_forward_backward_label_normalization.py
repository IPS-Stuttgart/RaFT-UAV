from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
)


def test_pair_forward_backward_normalizes_source_and_branch_labels() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq", "seq"],
            "time_s": [0.0, 1.0, 1.0],
            "source": [" RF ", "rf", "radar"],
            "candidate_branch": [" RAW ", "raw", "translated"],
            "track_id": ["start", "matching", "switched"],
            "x_m": [0.0, 0.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [0.0, 0.0, 0.0],
            "predicted_sigma_m": [1.0, 1.0, 1.0],
        }
    )
    config = CandidatePairForwardBackwardConfig(
        score_column="ranker_score",
        score_normalization="none",
        score_weight=0.0,
        sigma_log_weight=0.0,
        transition_distance_std_m=1.0,
        transition_speed_std_mps=0.0,
        max_speed_mps=100.0,
        speed_gate_penalty=0.0,
        acceleration_std_mps2=1.0,
        max_acceleration_mps2=100.0,
        acceleration_gate_penalty=0.0,
        source_switch_penalty=2.0,
        branch_switch_penalty=3.0,
        track_continuation_bonus=0.0,
        time_gap_penalty=0.0,
    )

    augmented = attach_pair_forward_backward_candidate_prior(candidates, config=config).rows
    final = augmented.loc[augmented["time_s"] == 1.0].set_index("track_id")

    matching = float(final.loc["matching", "candidate_pair_forward_backward_score"])
    switched = float(final.loc["switched", "candidate_pair_forward_backward_score"])
    assert matching > 0.99
    assert switched < 0.01
