from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluator import validate_mmaud_results_frame


def test_official_track5_results_loader_accepts_padded_headers() -> None:
    frame = validate_mmaud_results_frame(
        pd.DataFrame(
            {
                " Sequence ": ["001"],
                " Timestamp ": ["1706255054.386069"],
                " Position ": ["(1.5,2.5,3.5)"],
                " Classification ": ["2"],
            }
        )
    )

    assert frame.loc[0, "sequence_id"] == "001"
    assert frame.loc[0, "timestamp"] == 1706255054.386069
    assert frame.loc[0, ["x", "y", "z"]].tolist() == [1.5, 2.5, 3.5]
    assert frame.loc[0, "uav_type"] == "2"
