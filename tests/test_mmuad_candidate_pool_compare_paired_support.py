from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.candidate_pool_compare import _pooled_summary


def test_candidate_pool_compare_mse_uses_common_finite_support() -> None:
    frame_rows = pd.DataFrame(
        {
            "pool_label": ["partial", "partial"],
            "reference_oracle_all_3d_m": [1.0, 100.0],
            "candidate_oracle_all_3d_m": [2.0, float("nan")],
            "pool_frame_present": [True, False],
        }
    )

    summary = _pooled_summary(frame_rows, top_k_values=()).iloc[0]

    assert summary["pool_frame_present_fraction"] == 0.5
    assert summary["oracle_all_paired_frame_count"] == 1
    assert summary["reference_oracle_all_mse"] == 1.0
    assert summary["candidate_oracle_all_mse"] == 4.0
    assert summary["oracle_all_mse_delta"] == 3.0
