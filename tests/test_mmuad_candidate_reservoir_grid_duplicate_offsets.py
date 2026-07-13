from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_grid import run_candidate_reservoir_offset_grid


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "source": ["lidar"],
            "track_id": ["candidate"],
            "candidate_branch": ["raw"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "ranker_score": [1.0],
            "confidence": [1.0],
        }
    )


@pytest.mark.parametrize(
    ("argument", "specs", "duplicate_name"),
    [
        ("branch_offset_grid", ["raw=0,1", "raw=2"], "raw"),
        ("source_offset_grid", ["lidar=0", "lidar=1"], "lidar"),
    ],
)
def test_reservoir_grid_rejects_duplicate_offset_names(
    argument: str,
    specs: list[str],
    duplicate_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"offset grid specs contain duplicate names: '{duplicate_name}'",
    ):
        run_candidate_reservoir_offset_grid(
            _candidate_rows(),
            **{argument: specs},
        )
