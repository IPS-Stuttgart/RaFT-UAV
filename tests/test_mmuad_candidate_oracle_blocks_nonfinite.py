from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_oracle_blocks import (
    build_candidate_oracle_block_tables,
)


def test_candidate_oracle_blocks_treat_nonfinite_errors_as_missing() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 0.5],
            "oracle_all_3d_m": [float("nan"), float("-inf")],
            "oracle_all_rank": [4, 1],
            "oracle_in_top3": [False, True],
        }
    )

    blocks, summary = build_candidate_oracle_block_tables(
        rows,
        oracle_error_threshold_m=5.0,
        top_k=3,
        max_gap_s=1.0,
    )

    assert len(blocks) == 1
    block = blocks.iloc[0]
    assert block["oracle_failure_mode"] == "missing_good_candidate"
    assert int(block["frame_count"]) == 2
    assert pd.isna(block["oracle_all_3d_m_max"])

    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["oracle_failure_mode"] == "missing_good_candidate")
    ].iloc[0]
    assert int(pooled["frame_count"]) == 2
    assert pd.isna(pooled["block_max_error_m_max"])
