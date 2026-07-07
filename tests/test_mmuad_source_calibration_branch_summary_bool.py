from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.source_calibration_branches import source_calibration_branch_summary


def test_branch_summary_parses_string_boolean_flags() -> None:
    summary = source_calibration_branch_summary(
        pd.DataFrame(
            {
                "candidate_branch": ["raw", "calibrated", "raw"],
                "mmuad_candidate_branch_is_calibrated": ["False", "True", "0"],
                "mmuad_source_calibration_applied": ["False", "true", "1"],
            }
        )
    )

    assert summary["raw_branch_row_count"] == 2
    assert summary["calibrated_branch_row_count"] == 1
    assert summary["calibration_applied_row_count"] == 2
