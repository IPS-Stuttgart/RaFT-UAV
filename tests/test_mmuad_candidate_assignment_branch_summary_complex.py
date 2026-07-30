from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_assignment_branch_summary import (
    build_candidate_assignment_branch_summary,
)


def test_branch_summary_ignores_complex_numeric_measurements() -> None:
    frame_rows = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "state_error_3d_m": [1.0 + 2.0j, 3.0 + 0.0j],
            "oracle_error_3d_m": [2.0 + 3.0j, 2.0 + 0.0j],
            "dominant_error_3d_m": [4.0 + 5.0j, 4.0 + 0.0j],
            "state_regret_m": [6.0 + 7.0j, 5.0 + 0.0j],
            "dominant_regret_m": [8.0 + 9.0j, 6.0 + 0.0j],
            "oracle_mixture_weight": [0.1 + 0.2j, 0.75 + 0.0j],
            "oracle_weight_rank": [9.0 + 1.0j, 2.0 + 0.0j],
            "candidate_count": [10.0 + 3.0j, 7.0 + 0.0j],
            "dominant_is_oracle": [False, True],
            "oracle_in_topk_by_weight": [False, True],
        }
    )

    summary = build_candidate_assignment_branch_summary(frame_rows)
    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["group_label"] == "__all__")
    ].iloc[0]

    assert pooled["state_error_3d_m_mse"] == pytest.approx(9.0)
    assert pooled["oracle_error_3d_m_mse"] == pytest.approx(4.0)
    assert pooled["dominant_error_3d_m_mse"] == pytest.approx(16.0)
    assert pooled["state_regret_m_mean"] == pytest.approx(5.0)
    assert pooled["dominant_regret_m_mean"] == pytest.approx(6.0)
    assert pooled["oracle_mixture_weight_mean"] == pytest.approx(0.75)
    assert pooled["oracle_weight_rank_p50"] == pytest.approx(2.0)
    assert pooled["candidate_count_mean"] == pytest.approx(7.0)
