from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_source_branch_reservoir import (
    source_branch_reservoir_summary,
)


def test_source_branch_summary_parses_serialized_selection_flags() -> None:
    flags = [
        False,
        True,
        "False",
        "true",
        "0",
        "1",
        "no",
        "yes",
        "",
        "null",
        0,
        2,
        np.nan,
    ]
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq"] * len(flags),
            "time_s": np.arange(len(flags), dtype=float),
            "source": ["lidar"] * len(flags),
            "candidate_branch": ["raw"] * len(flags),
            "candidate_reservoir_reason": ["branch:raw"] * len(flags),
            "candidate_reservoir_reason_count": [1] * len(flags),
            "candidate_reservoir_protected": [False] * len(flags),
            "candidate_source_branch_quota_selected": flags,
            "candidate_source_branch_diversity_selected": list(reversed(flags)),
        }
    )

    summary = source_branch_reservoir_summary(rows, rows)

    assert summary["source_branch_quota_selected_rows"] == 5
    assert summary["source_branch_diversity_selected_rows"] == 5
