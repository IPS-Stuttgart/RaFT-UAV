from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def test_template_snap_rejects_out_of_range_classification_ids() -> None:
    results = pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [4],
        }
    )
    template = pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [0.0]})

    with pytest.raises(ValueError, match="must be one of"):
        snap_official_results_to_template(results, template)
