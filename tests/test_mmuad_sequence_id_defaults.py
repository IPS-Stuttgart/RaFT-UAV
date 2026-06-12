from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.io import load_candidate_csv, load_truth_csv


def test_candidate_loader_fills_blank_and_missing_sequence_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "sequence_id": ["explicit", np.nan, ""],
            "time_s": [0.0, 1.0, 2.0],
            "source": ["radar", "radar", "radar"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)

    frame = load_candidate_csv(path, default_sequence_id="seq_from_folder")

    by_time = frame.rows.set_index("time_s")["sequence_id"].to_dict()
    assert by_time[0.0] == "explicit"
    assert by_time[1.0] == "seq_from_folder"
    assert by_time[2.0] == "seq_from_folder"
    assert "nan" not in set(frame.rows["sequence_id"])
    assert "" not in set(frame.rows["sequence_id"])


def test_truth_loader_fills_blank_and_missing_sequence_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "truth.csv"
    pd.DataFrame(
        {
            "sequence_id": [np.nan, "", "explicit_truth"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)

    frame = load_truth_csv(path, default_sequence_id="seq_from_folder")

    by_time = frame.rows.set_index("time_s")["sequence_id"].to_dict()
    assert by_time[0.0] == "seq_from_folder"
    assert by_time[1.0] == "seq_from_folder"
    assert by_time[2.0] == "explicit_truth"
    assert "nan" not in set(frame.rows["sequence_id"])
    assert "" not in set(frame.rows["sequence_id"])
