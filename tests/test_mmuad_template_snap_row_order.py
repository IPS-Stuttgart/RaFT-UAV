from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def test_template_snap_preserves_requested_row_order() -> None:
    results = pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq002", "seq002"],
            "Timestamp": [0.0, 10.0, 1.0, 5.0],
            "Position": ["(0,0,0)", "(10,10,10)", "(1,1,1)", "(5,5,5)"],
            "Classification": [1, 1, 2, 2],
        }
    )
    template = pd.DataFrame(
        {
            "Sequence": ["seq002", "seq001", "seq002", "seq001"],
            "Timestamp": [5.0, 10.0, 1.0, 0.0],
        }
    )

    snapped, diagnostics = snap_official_results_to_template(
        results,
        template,
        resample_method="nearest",
    )

    expected_keys = list(
        template[["Sequence", "Timestamp"]].itertuples(index=False, name=None)
    )
    snapped_keys = list(
        snapped[["Sequence", "Timestamp"]].itertuples(index=False, name=None)
    )
    diagnostic_keys = list(
        diagnostics[["Sequence", "Timestamp"]].itertuples(index=False, name=None)
    )

    assert snapped_keys == expected_keys
    assert diagnostic_keys == expected_keys
    assert diagnostics["template_row_index"].tolist() == [0, 1, 2, 3]
