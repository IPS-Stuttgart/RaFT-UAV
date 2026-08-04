from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_gap import build_candidate_oracle_gap


def test_candidate_oracle_gap_uses_final_duplicate_truth_sample() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "source": ["radar"],
            "track_id": ["candidate"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "confidence": [1.0],
        }
    )
    selected = candidates.copy()
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": ["0", 0.0, 1.0],
            "x_m": [100.0, 0.0, 1.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )

    rows = build_candidate_oracle_gap(
        candidates,
        selected,
        truth,
        max_time_delta_s=0.1,
    )

    at_zero = rows.loc[rows["time_s"] == 0.0]
    assert len(at_zero) == 1
    row = at_zero.iloc[0]
    assert row["truth_x_m"] == pytest.approx(0.0)
    assert row["selected_minus_truth_error_m"] == pytest.approx(0.0)
    assert row["nearest_minus_truth_error_m"] == pytest.approx(0.0)
    assert row["candidate_regret_m"] == pytest.approx(0.0)
