from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.evaluate import load_submission_csv
from raft_uav.mmuad.evaluator import load_mmaud_results_csv


def test_load_submission_csv_preserves_zero_padded_sequence_ids(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    path.write_text(
        "\n".join(
            [
                "Sequence,time_s,x_m,y_m,z_m,track_id",
                "001,0.0,1.0,2.0,3.0,uav-a",
                "010,1.0,4.0,5.0,6.0,uav-b",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_submission_csv(path)

    assert rows["sequence_id"].tolist() == ["001", "010"]
    assert rows["track_id"].tolist() == ["uav-a", "uav-b"]
    assert rows["time_s"].tolist() == [0.0, 1.0]


def test_official_track5_loader_accepts_whitespace_padded_headers(tmp_path: Path) -> None:
    results_path = tmp_path / "mmaud_results.csv"
    pd.DataFrame(
        {
            " Sequence ": ["seq0"],
            " Timestamp ": ["1.0"],
            " Position ": [(1.0, 2.0, 3.0)],
            " Classification ": ["2"],
        }
    ).to_csv(results_path, index=False)

    rows = load_mmaud_results_csv(results_path).rows

    assert rows.to_dict("records") == [
        {
            "sequence_id": "seq0",
            "timestamp": 1.0,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "uav_type": "2",
            "score": 1.0,
        }
    ]
