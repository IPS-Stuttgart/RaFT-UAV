from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_consensus_quota import (
    _cap_with_mandatory_consensus,
    consensus_quota_summary,
)


def _summary_rows() -> pd.DataFrame:
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
    count = len(flags)
    return pd.DataFrame(
        {
            "sequence_id": ["seq"] * count,
            "time_s": np.arange(count, dtype=float),
            "source": ["radar"] * count,
            "candidate_branch": ["raw"] * count,
            "track_id": [f"candidate-{index}" for index in range(count)],
            "x_m": np.arange(count, dtype=float),
            "y_m": np.zeros(count),
            "z_m": np.ones(count),
            "candidate_reservoir_reason": ["global_top_n"] * count,
            "candidate_reservoir_score": np.arange(count, dtype=float),
            "candidate_reservoir_protected": [False] * count,
            "candidate_consensus_quota_selected": flags,
            "candidate_consensus_supported": list(reversed(flags)),
        }
    )


def test_consensus_quota_summary_parses_serialized_boolean_flags() -> None:
    rows = _summary_rows()

    summary = consensus_quota_summary(rows, rows)

    assert summary["consensus_quota_selected_rows"] == 5
    assert summary["consensus_supported_output_rows"] == 5
    assert summary["consensus_quota_selected_frame_count"] == 5


def test_frame_cap_does_not_make_serialized_false_flags_mandatory() -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 0.0],
            "source": ["radar", "radar"],
            "track_id": ["serialized-false", "serialized-true"],
            "candidate_reservoir_score": [10.0, 1.0],
            "candidate_reservoir_reason": [
                "global_top_n",
                "consensus:cross_source",
            ],
            "candidate_consensus_quota_selected": ["False", "True"],
            "candidate_consensus_quota_rank": [0.0, 1.0],
        }
    )

    capped = _cap_with_mandatory_consensus(
        rows,
        max_candidates_per_frame=1,
        cap_reason_bonus=0.0,
        preserve_reason_prefixes=("consensus:",),
    )

    assert capped["track_id"].tolist() == ["serialized-true"]
