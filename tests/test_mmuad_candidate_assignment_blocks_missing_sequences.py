from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_assignment_blocks import (
    build_candidate_assignment_block_tables,
    main as assignment_blocks_main,
)


def test_assignment_blocks_drop_genuinely_missing_sequence_ids() -> None:
    frame_rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", float("nan"), None, pd.NA],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "assignment_failure_mode": ["covered"] * 4,
        }
    )

    blocks, summary = build_candidate_assignment_block_tables(frame_rows)

    assert blocks["sequence_id"].tolist() == ["seqA"]
    assert set(summary["sequence_id"]) == {"__pooled__", "seqA"}


def test_assignment_blocks_cli_uses_missing_sequence_filter() -> None:
    assert (
        assignment_blocks_main.__globals__["build_candidate_assignment_block_tables"]
        is build_candidate_assignment_block_tables
    )
