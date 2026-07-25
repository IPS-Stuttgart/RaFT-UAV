from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration_branches import (
    build_source_calibration_branch_union,
)


def _candidate_frame() -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq"],
                "time_s": [0.0],
                "source": ["lidar"],
                "track_id": ["track"],
                "x_m": [1.0],
                "y_m": [2.0],
                "z_m": [3.0],
                "confidence": [0.9],
            }
        )
    )


@pytest.mark.parametrize(
    ("raw_branch", "calibrated_branch"),
    [
        ("shared", "shared"),
        ("shared label", "shared_label"),
        ("RAW", "raw"),
    ],
)
def test_source_calibration_union_rejects_colliding_branch_labels(
    raw_branch: str,
    calibrated_branch: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="raw and calibrated candidate branch labels must be distinct",
    ):
        build_source_calibration_branch_union(
            _candidate_frame(),
            {"mode": "identity"},
            raw_branch=raw_branch,
            calibrated_branch=calibrated_branch,
        )
