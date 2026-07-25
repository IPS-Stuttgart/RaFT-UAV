from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.track5_template_resample import (
    summarize_template_resample_diagnostics,
)


def test_template_resample_summary_parses_csv_float_boolean_flags() -> None:
    diagnostics = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "valid": [1.0, 0.0],
            "extrapolated": ["1.0", "0.0"],
            "large_gap_fallback": pd.Series([0.0, 1.0], dtype="Float64"),
            "resample_method": ["linear", "nearest"],
            "source_row_count": [2, 2],
            "nearest_time_delta_s": [0.0, 1.0],
            "interpolation_gap_s": [0.0, 5.0],
            "classification_source": ["sequence-mode", "nearest"],
        }
    )

    summary = summarize_template_resample_diagnostics(diagnostics).iloc[0]

    assert summary["valid_row_count"] == 1
    assert summary["invalid_row_count"] == 1
    assert summary["extrapolated_row_count"] == 1
    assert summary["large_gap_fallback_row_count"] == 1
