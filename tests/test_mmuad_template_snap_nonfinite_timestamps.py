from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq001"],
            "Timestamp": [0.0, np.inf, -np.inf],
            "Position": ["(1,2,3)", "(4,5,6)", "(7,8,9)"],
            "Classification": [2, 2, 2],
        }
    )


def test_template_snap_drops_nonfinite_template_timestamps() -> None:
    template = pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq001"],
            "Timestamp": [0.0, np.inf, -np.inf],
        }
    )

    snapped, diagnostics = snap_official_results_to_template(_results(), template)

    assert snapped["Timestamp"].tolist() == [0.0]
    assert diagnostics["Timestamp"].tolist() == [0.0]


def test_template_snap_drops_nonfinite_source_timestamps() -> None:
    template = pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [0.0]})

    snapped, diagnostics = snap_official_results_to_template(_results(), template)

    assert len(snapped) == 1
    assert int(diagnostics.loc[0, "source_row_count"]) == 1
