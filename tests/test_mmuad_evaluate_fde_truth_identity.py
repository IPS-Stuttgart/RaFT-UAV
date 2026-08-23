import pandas as pd

from raft_uav.mmuad.evaluate import metrics_from_matches


def test_fde_keeps_disjoint_truth_tracks_separate():
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1", "seq1"],
            "track_id": ["a", "a", "b", "b"],
            "truth_track_id": ["a", "a", "b", "b"],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "truth_time_s": [0.0, 1.0, 2.0, 3.0],
            "matched": [True, True, True, True],
            "error_3d_m": [1.0, 10.0, 2.0, 30.0],
            "error_2d_m": [1.0, 6.0, 2.0, 18.0],
        }
    )
    submission = matches[["sequence_id", "track_id", "time_s"]].copy()
    truth = matches[["sequence_id", "truth_track_id", "truth_time_s"]].rename(
        columns={"truth_track_id": "track_id", "truth_time_s": "time_s"}
    )

    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["fde_3d_m"] == 20.0
    assert metrics["pooled"]["fde_2d_m"] == 12.0
    assert metrics["sequences"]["seq1"]["fde_3d_m"] == 20.0
    assert metrics["sequences"]["seq1"]["fde_2d_m"] == 12.0


def test_fde_preserves_anonymous_truth_handoff_fallback():
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1", "seq1"],
            "track_id": ["first", "first", "second", "second"],
            "truth_track_id": ["", "", "", ""],
            "time_s": [0.0, 1.0, 2.0, 3.0],
            "truth_time_s": [0.0, 1.0, 2.0, 3.0],
            "matched": [True, True, True, True],
            "error_3d_m": [1.0, 10.0, 2.0, 30.0],
            "error_2d_m": [1.0, 6.0, 2.0, 18.0],
        }
    )
    submission = matches[["sequence_id", "track_id", "time_s"]].copy()
    truth = matches[["sequence_id", "truth_time_s"]].rename(
        columns={"truth_time_s": "time_s"}
    )

    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["fde_3d_m"] == 30.0
    assert metrics["pooled"]["fde_2d_m"] == 18.0
