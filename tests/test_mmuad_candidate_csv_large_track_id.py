from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.io import load_candidate_csv


def test_candidate_csv_preserves_plain_numeric_track_ids_beyond_uint64(
    tmp_path: Path,
) -> None:
    huge_track_id = 2**80 + 123
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq", "seq"],
            "time_s": [0.0, 1.0],
            "track_id": ["7", str(huge_track_id)],
            "x_m": [1.0, 2.0],
            "y_m": [3.0, 4.0],
            "z_m": [5.0, 6.0],
        }
    ).to_csv(path, index=False)

    candidates = load_candidate_csv(path)

    assert candidates.rows["track_id"].tolist() == [7, huge_track_id]


def test_candidate_csv_restores_each_numeric_identifier_column(tmp_path: Path) -> None:
    huge_object_id = 2**80 + 456
    path = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "sequence_id": ["seq"],
            "time_s": [0.0],
            "track_id": ["opaque-track"],
            "object_id": [str(huge_object_id)],
            "cluster_id": ["17"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    ).to_csv(path, index=False)

    candidates = load_candidate_csv(path)

    row = candidates.rows.iloc[0]
    assert row["track_id"] == "opaque-track"
    assert row["object_id"] == huge_object_id
    assert row["cluster_id"] == 17
