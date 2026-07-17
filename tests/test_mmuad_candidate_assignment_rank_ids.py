from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_assignment_diagnostics import (
    _safe_int,
    build_candidate_assignment_diagnostics,
)


def test_candidate_assignment_diagnostics_preserves_large_integer_rank() -> None:
    assignments = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "candidate_rank": ["9007199254740993"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    frames, _ = build_candidate_assignment_diagnostics(assignments, truth)

    assert frames.loc[0, "oracle_candidate_rank"] == 9007199254740993
    assert frames.loc[0, "dominant_candidate_rank"] == 9007199254740993


@pytest.mark.parametrize("value", [7.5, "7.5", True, np.array([7])])
def test_candidate_assignment_rank_rejects_non_integer_values(value: object) -> None:
    assert _safe_int(value) is None
