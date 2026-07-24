from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_assignment_blocks import (
    build_candidate_assignment_block_tables,
)


def test_assignment_blocks_drop_nonfinite_timestamps() -> None:
    frame_rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [-np.inf, 0.0, np.inf],
            "assignment_failure_mode": ["covered", "covered", "covered"],
        }
    )

    blocks, summary = build_candidate_assignment_block_tables(frame_rows)

    assert len(blocks) == 1
    assert blocks["frame_count"].tolist() == [1]
    assert blocks["start_time_s"].tolist() == [0.0]
    assert blocks["end_time_s"].tolist() == [0.0]
    assert np.isfinite(
        blocks[["start_time_s", "end_time_s", "duration_s"]].to_numpy(float)
    ).all()
    assert int(summary.loc[summary["sequence_id"] == "__pooled__", "frame_count"].iloc[0]) == 1
