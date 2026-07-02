from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad import run
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def test_track5_submission_ensemble_accepts_numpy_array_position_repr(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["array([1.5, 2.5, 3.5])"],
            "Classification": [2],
        }
    ).to_csv(path, index=False)

    loaded = load_track5_submission(path)

    assert loaded.loc[0, ["state_x_m", "state_y_m", "state_z_m"]].tolist() == [
        1.5,
        2.5,
        3.5,
    ]


def test_mmuad_run_suffix_like_value_option_consumes_next_argument() -> None:
    assert run._option_consumes_next("--not-a-bool-only") is True
