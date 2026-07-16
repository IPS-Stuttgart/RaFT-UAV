from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import build_reservoir_summary


def test_summary_parses_serialized_candidate_protection_flags() -> None:
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
            "candidate_reservoir_protected": flags,
        }
    )

    summary = build_reservoir_summary(rows, rows)

    assert summary["reservoir_protected_count"] == 5
