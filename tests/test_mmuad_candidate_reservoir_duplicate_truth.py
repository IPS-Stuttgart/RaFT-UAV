from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables


def test_candidate_reservoir_oracle_uses_final_duplicate_truth_sample() -> None:
    reservoir = pd.DataFrame(
        {
            "sequence_id": ["sequence-a"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["7"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [5.0],
            "candidate_reservoir_score": [1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["sequence-a", "sequence-a", "sequence-a"],
            "time_s": [0.0, 0.0, 1.0],
            "x_m": [100.0, 0.0, 1.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [5.0, 5.0, 5.0],
        }
    )

    frame_rows, pooled, by_sequence = build_oracle_recall_tables(
        reservoir,
        truth,
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
    )

    assert frame_rows.loc[0, "oracle_all_3d_m"] == pytest.approx(0.0)
    assert frame_rows.loc[0, "oracle_top1_3d_m"] == pytest.approx(0.0)
    assert pooled.loc[0, "oracle_all_3d_m_rmse"] == pytest.approx(0.0)
    assert by_sequence.loc[0, "oracle_all_3d_m_rmse"] == pytest.approx(0.0)
