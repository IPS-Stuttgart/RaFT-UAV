from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
)


def test_candidate_oracle_attribution_uses_final_duplicate_truth_row() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["candidate"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
            "candidate_reservoir_score": [1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [1.0, "0", 0.0, 2.0],
            "x_m": [1.0, 100.0, 0.0, 2.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
        }
    )

    frame_rows, pooled, _by_branch, _by_source = (
        build_candidate_oracle_attribution_tables(
            candidates,
            truth,
            top_k_values=(1,),
            max_truth_time_delta_s=0.0,
        )
    )

    assert len(frame_rows) == 1
    assert frame_rows.loc[0, "truth_time_delta_s"] == 0.0
    assert frame_rows.loc[0, "oracle_all_3d_m"] == 0.0
    assert pooled.loc[0, "oracle_all_3d_m_mse"] == 0.0
