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


def test_candidate_loader_fills_blank_sources_and_nulls_blank_track_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "source": ["radar", np.nan, ""],
            "track_id": ["track-a", np.nan, ""],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    ).to_csv(path, index=False)

    frame = load_candidate_csv(path, source="radar_default")

    by_time = frame.rows.set_index("time_s")
    assert by_time.loc[0.0, "source"] == "radar"
    assert by_time.loc[1.0, "source"] == "radar_default"
    assert by_time.loc[2.0, "source"] == "radar_default"
    assert by_time.loc[0.0, "track_id"] == "track-a"
    assert pd.isna(by_time.loc[1.0, "track_id"])
    assert pd.isna(by_time.loc[2.0, "track_id"])


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
