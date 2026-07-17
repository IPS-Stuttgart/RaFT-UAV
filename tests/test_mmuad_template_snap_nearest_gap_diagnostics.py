from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def test_nearest_resampling_does_not_report_large_gap_fallback() -> None:
    results = pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [0.0, 10.0],
            "Position": ["(0,0,0)", "(10,20,2)"],
            "Classification": [2, 2],
        }
    )
    template = pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [5.0]})

    snapped, diagnostics = snap_official_results_to_template(
        results,
        template,
        resample_method="nearest",
        max_interpolation_gap_s=0.0,
    )

    assert snapped.loc[0, "Position"] == "(0,0,0)"
    assert diagnostics.loc[0, "method"] == "nearest"
    assert bool(diagnostics.loc[0, "large_gap_fallback"]) is False
    assert float(diagnostics.loc[0, "interpolation_gap_s"]) == 10.0
