from pathlib import Path

import pandas as pd

from raft_uav.mmuad.evaluate import load_submission_csv


def test_load_submission_csv_accepts_case_insensitive_alias_columns(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Sequence_ID": ["seqA"],
            "Time_S": [1.25],
            "Track": ["track7"],
            "X": [10.0],
            "Y": [20.0],
            "Z": [30.0],
            "Score": [0.8],
        }
    ).to_csv(path, index=False)

    frame = load_submission_csv(path)

    assert frame.loc[0, "sequence_id"] == "seqA"
    assert frame.loc[0, "time_s"] == 1.25
    assert frame.loc[0, "track_id"] == "track7"
    assert frame.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [10.0, 20.0, 30.0]
    assert frame.loc[0, "score"] == 0.8
