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


def test_structured_numpy_candidate_table_preserves_named_fields(tmp_path) -> None:
    path = tmp_path / "structured_candidates.npy"
    dtype = np.dtype(
        [
            ("time_s", "f8"),
            ("x_m", "f8"),
            ("y_m", "f8"),
            ("z_m", "f8"),
            ("confidence", "f8"),
        ]
    )
    np.save(
        path,
        np.array(
            [
                (1.5, 1.0, 2.0, 3.0, 0.9),
                (2.5, 4.0, 5.0, 6.0, 0.8),
            ],
            dtype=dtype,
        ),
    )

    frame = load_candidate_file(path, default_sequence_id="seq-c", source="numpy").rows

    assert frame[["time_s", "x_m", "y_m", "z_m", "confidence"]].to_dict("records") == [
        {"time_s": 1.5, "x_m": 1.0, "y_m": 2.0, "z_m": 3.0, "confidence": 0.9},
        {"time_s": 2.5, "x_m": 4.0, "y_m": 5.0, "z_m": 6.0, "confidence": 0.8},
    ]


def test_npz_trajectory_preferred_key_matching_is_case_insensitive(tmp_path) -> None:
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        Metadata=np.array([10.0, 20.0]),
        Trajectory=np.array([[8.0, -1.0, 2.5, 4.0, 0.7]], dtype=float),
    )

    frame = load_candidate_file(path, default_sequence_id="seq-d", source="numpy").rows

    assert frame[
        ["sequence_id", "time_s", "source", "x_m", "y_m", "z_m", "confidence"]
    ].to_dict("records") == [
        {
            "sequence_id": "seq-d",
            "time_s": 8.0,
            "source": "numpy",
            "x_m": -1.0,
            "y_m": 2.5,
            "z_m": 4.0,
            "confidence": 0.7,
        }
    ]


def test_npz_trajectory_exact_key_wins_over_case_insensitive_alias(tmp_path) -> None:
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        trajectory=np.array([[9.0, 1.0, 2.0, 3.0, 0.8]], dtype=float),
        Trajectory=np.array([10.0, 20.0]),
    )

    frame = load_candidate_file(path, default_sequence_id="seq-e", source="numpy").rows

    assert frame[
        ["sequence_id", "time_s", "source", "x_m", "y_m", "z_m", "confidence"]
    ].to_dict("records") == [
        {
            "sequence_id": "seq-e",
            "time_s": 9.0,
            "source": "numpy",
            "x_m": 1.0,
            "y_m": 2.0,
            "z_m": 3.0,
            "confidence": 0.8,
        }
    ]


def test_npz_trajectory_rejects_ambiguous_case_insensitive_aliases(tmp_path) -> None:
    path = tmp_path / "candidate.npz"
    np.savez(
        path,
        Trajectory=np.array([[9.0, 1.0, 2.0, 3.0, 0.8]], dtype=float),
        TRAJECTORY=np.array([[10.0, 4.0, 5.0, 6.0, 0.7]], dtype=float),
    )

    with pytest.raises(ValueError, match="ambiguous arrays for preferred key 'trajectory'"):
        load_candidate_file(path, default_sequence_id="seq-f", source="numpy")
