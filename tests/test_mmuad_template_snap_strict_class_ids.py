from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_utils import load_official_track5_results_frame_from_frame


def test_template_snap_rejects_near_integer_classification_labels() -> None:
    rows = pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": ["1.000001"],
        }
    )

    with pytest.raises(ValueError, match="integer ids"):
        load_official_track5_results_frame_from_frame(rows)
