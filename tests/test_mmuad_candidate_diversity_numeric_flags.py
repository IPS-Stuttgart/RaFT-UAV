from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_diversity import diversify_candidate_reservoir


def _rows(flags: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [1.0, 1.0],
            "track_id": ["best", "protected"],
            "x_m": [0.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "candidate_reservoir_score": [1.0, 0.1],
            "candidate_reservoir_protected": flags,
        }
    )


@pytest.mark.parametrize(
    "invalid_flag",
    [2, -1, 0.5, np.int64(2), np.float64(-0.25)],
)
def test_diversity_rejects_nonbinary_numeric_protection_flags(
    invalid_flag: object,
) -> None:
    with pytest.raises(ValueError, match="candidate_reservoir_protected"):
        diversify_candidate_reservoir(
            _rows([0, invalid_flag]),
            radius_m=0.0,
            max_candidates_per_frame=1,
        )


def test_diversity_preserves_binary_numeric_protection_flags() -> None:
    output = diversify_candidate_reservoir(
        _rows([0, 1]),
        radius_m=0.0,
        max_candidates_per_frame=1,
    )

    assert output["track_id"].tolist() == ["protected"]
