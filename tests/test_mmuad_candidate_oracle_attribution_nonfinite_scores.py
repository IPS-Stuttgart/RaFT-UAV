from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
)


def test_oracle_attribution_falls_back_from_nonfinite_scores() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.0],
            "source": ["oracle", "distractor"],
            "track_id": ["oracle", "distractor"],
            "candidate_branch": ["raw", "translated"],
            "x_m": [0.0, 20.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
            "candidate_reservoir_score": [np.nan, np.inf],
            "ranker_score": [0.9, 0.1],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    )

    frame_rows, pooled, _by_branch, _by_source = (
        build_candidate_oracle_attribution_tables(
            candidates,
            truth,
            top_k_values=(1,),
            max_truth_time_delta_s=0.1,
        )
    )

    assert frame_rows.loc[0, "oracle_all_rank"] == 1
    assert frame_rows.loc[0, "oracle_all_candidate_score"] == 0.9
    assert frame_rows.loc[0, "oracle_top1_3d_m"] == 0.0
    assert pooled.loc[0, "oracle_in_top1_fraction"] == 1.0
