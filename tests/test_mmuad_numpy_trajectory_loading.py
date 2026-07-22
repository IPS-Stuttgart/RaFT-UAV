from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.io import load_candidate_file, load_truth_file


def test_single_row_numpy_truth_table_preserves_explicit_timestamp(tmp_path) -> None:
    path = tmp_path / "trajectory.npy"
    np.save(path, np.array([[12.5, 1.0, 2.0, 3.0]], dtype=float))

    frame = load_truth_file(path, default_sequence_id="seq-a").rows

    assert frame[["sequence_id", "time_s", "x_m", "y_m", "z_m"]].to_dict("records") == [
        {"sequence_id": "seq-a", "time_s": 12.5, "x_m": 1.0, "y_m": 2.0, "z_m": 3.0}
    ]


def test_single_row_numpy_candidate_table_preserves_time_and_confidence(tmp_path) -> None:
    path = tmp_path / "candidate.npy"
    np.save(path, np.array([[8.0, -1.0, 2.5, 4.0, 0.7]], dtype=float))

    frame = load_candidate_file(path, default_sequence_id="seq-b", source="numpy").rows

    assert frame[
        ["sequence_id", "time_s", "source", "x_m", "y_m", "z_m", "confidence"]
    ].to_dict("records") == [
        {
            "sequence_id": "seq-b",
            "time_s": 8.0,
            "source": "numpy",
            "x_m": -1.0,
            "y_m": 2.5,
            "z_m": 4.0,
            "confidence": 0.7,
        }
    ]


def test_numpy_trajectory_column_vector_is_rejected(tmp_path) -> None:
    path = tmp_path / "column-vector.npy"
    np.save(path, np.array([[1.0], [2.0], [3.0], [0.7]], dtype=float))

    with pytest.raises(ValueError, match=r"shape \(N, >=3\), got \(4, 1\)"):
        load_truth_file(path)
