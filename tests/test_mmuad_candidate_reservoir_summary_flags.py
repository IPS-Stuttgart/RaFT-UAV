from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import build_reservoir_summary


def _summary_rows(flags: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
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
    rows = _summary_rows(flags)

    summary = build_reservoir_summary(rows, rows)

    assert summary["reservoir_protected_count"] == 5


@pytest.mark.parametrize(
    "invalid_flag",
    [
        "maybe",
        1.0 + 2.0j,
        np.complex64(1.0 + 0.0j),
        np.array(1.0 + 2.0j, dtype=object),
        np.ma.array(1.0 + 0.0j, mask=False),
    ],
)
def test_summary_rejects_malformed_candidate_protection_flags(
    invalid_flag: object,
) -> None:
    rows = _summary_rows([invalid_flag])

    with pytest.raises(ValueError, match="candidate_reservoir_protected"):
        build_reservoir_summary(rows, rows)
