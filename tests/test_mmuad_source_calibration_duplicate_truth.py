from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import build_source_calibration_pairs


def test_source_calibration_uses_final_duplicate_truth_sample() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq001"],
                "time_s": [-0.1],
                "source": ["lidar_360"],
                "track_id": ["candidate"],
                "x_m": [0.0],
                "y_m": [0.0],
                "z_m": [0.0],
                "confidence": [1.0],
            }
        )
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": ["0", 0.0, 1.0],
            "x_m": [100.0, 0.0, 1.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
        }
    )

    pairs = build_source_calibration_pairs(
        candidates,
        truth,
        max_truth_time_delta_s=0.2,
        max_pair_distance_m=10.0,
    )

    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["truth_x_m"] == pytest.approx(0.0)
    assert row["pair_error_before_m"] == pytest.approx(0.0)
