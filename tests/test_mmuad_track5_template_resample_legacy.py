from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import summarize_template_resample_diagnostics


def test_template_resample_summary_accepts_legacy_optional_columns() -> None:
    diagnostics = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "nearest_time_delta_s": [0.0, -2.0],
            "valid": [True, False],
            "extrapolated": [False, True],
        }
    )

    summary = summarize_template_resample_diagnostics(diagnostics).set_index("sequence_id")

    assert summary.loc["seq0001", "template_row_count"] == 2
    assert summary.loc["seq0001", "invalid_row_count"] == 1
    assert summary.loc["seq0001", "linear_method_row_count"] == 2
    assert summary.loc["seq0001", "nearest_method_row_count"] == 0
    assert summary.loc["seq0001", "source_row_count_min"] == 0
    assert summary.loc["seq0001", "source_row_count_max"] == 0
    assert summary.loc["seq0001", "nearest_time_delta_abs_max_s"] == pytest.approx(2.0)
