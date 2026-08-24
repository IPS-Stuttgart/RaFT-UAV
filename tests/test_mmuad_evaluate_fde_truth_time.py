import pandas as pd

from raft_uav.mmuad.evaluate import metrics_from_matches


def test_submission_eval_fde_uses_latest_matched_truth_time():
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [9.9, 10.1],
            "track_id": ["A", "B"],
            "truth_time_s": [10.0, 9.8],
            "truth_track_id": ["A", "A"],
            "matched": [True, True],
            "error_3d_m": [5.0, 100.0],
            "error_2d_m": [3.0, 60.0],
        }
    )
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [9.9, 10.1],
            "track_id": ["A", "B"],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1"],
            "time_s": [10.0, 9.8],
            "track_id": ["A", "A"],
            "x_m": [0.0, 0.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )

    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["fde_3d_m"] == 5.0
    assert metrics["pooled"]["fde_2d_m"] == 3.0
    assert metrics["sequences"]["seq1"]["fde_3d_m"] == 5.0
    assert metrics["sequences"]["seq1"]["fde_2d_m"] == 3.0


def test_submission_eval_fde_falls_back_to_latest_submission_time():
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [2.0, 0.0, 1.0],
            "matched": [True, True, True],
            "error_3d_m": [20.0, 0.0, 10.0],
            "error_2d_m": [12.0, 0.0, 6.0],
        }
    )
    submission = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [2.0, 0.0, 1.0],
        }
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [0.0, 1.0, 2.0],
        }
    )

    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["fde_3d_m"] == 20.0
    assert metrics["pooled"]["fde_2d_m"] == 12.0
    assert metrics["sequences"]["seq1"]["fde_3d_m"] == 20.0
    assert metrics["sequences"]["seq1"]["fde_2d_m"] == 12.0
