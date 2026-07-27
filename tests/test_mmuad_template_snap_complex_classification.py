from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_utils import load_official_track5_results_frame_from_frame


@pytest.mark.parametrize(
    "classification",
    [
        1 + 2j,
        np.complex64(2 + 0j),
        np.asarray(3 + 4j),
    ],
)
def test_official_track5_results_reject_complex_classification(classification: object) -> None:
    frame = pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [0.0],
            "Position": ["(0,0,0)"],
            "Classification": pd.Series([classification], dtype=object),
        }
    )

    with pytest.raises(ValueError, match="not complex numbers"):
        load_official_track5_results_frame_from_frame(frame)
