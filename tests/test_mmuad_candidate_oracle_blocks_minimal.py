from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_blocks import build_candidate_oracle_block_tables


def test_candidate_oracle_blocks_accept_minimal_rows_without_optional_columns() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.5, 1.0],
            "oracle_all_3d_m": [0.5, 0.4, 0.6],
            "oracle_all_rank": [1, 8, 9],
        }
    )

    blocks, summary = build_candidate_oracle_block_tables(
        rows,
        oracle_error_threshold_m=5.0,
        top_k=3,
        max_gap_s=1.0,
    )

    assert not blocks.empty
    assert not summary.empty
    assert "good_candidate_buried" in set(blocks["oracle_failure_mode"])
    assert blocks["candidate_count_mean"].isna().all()
    assert blocks["dominant_oracle_branch"].eq("").all()
    assert blocks["dominant_oracle_source"].eq("").all()
