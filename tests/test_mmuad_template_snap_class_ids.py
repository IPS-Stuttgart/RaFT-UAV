from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.submission import OFFICIAL_TRACK5_CLASS_IDS
from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def test_template_snap_rejects_unofficial_class_labels() -> None:
    invalid_class_id = max(OFFICIAL_TRACK5_CLASS_IDS) + 1
    results = pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": [invalid_class_id],
        }
    )
    template = pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [0.0]})

    with pytest.raises(ValueError, match="must be one of"):
        snap_official_results_to_template(results, template)
