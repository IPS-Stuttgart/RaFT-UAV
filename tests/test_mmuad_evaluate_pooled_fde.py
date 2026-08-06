import pandas as pd

from raft_uav.mmuad.evaluate import metrics_from_matches


def test_pooled_fde_averages_each_sequence_endpoint():
    matches = pd.DataFrame(
        {
            "sequence_id": ["short", "short", "long", "long"],
            "time_s": [0.0, 1.0, 0.0, 2.0],
            "truth_time_s": [0.0, 1.0, 0.0, 2.0],
            "matched": [True, True, True, True],
            "error_3d_m": [1.0, 100.0, 2.0, 20.0],
            "error_2d_m": [1.0, 60.0, 2.0, 12.0],
        }
    )
    submission = matches[["sequence_id", "time_s"]].copy()
    truth = matches[["sequence_id", "truth_time_s"]].rename(
        columns={"truth_time_s": "time_s"}
    )

    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["fde_3d_m"] == 60.0
    assert metrics["pooled"]["fde_2d_m"] == 36.0
    assert metrics["sequences"]["short"]["fde_3d_m"] == 100.0
    assert metrics["sequences"]["long"]["fde_3d_m"] == 20.0


def test_sequence_fde_averages_each_track_endpoint():
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1", "seq1"],
            "track_id": ["a", "a", "b", "b"],
            "truth_track_id": ["a", "a", "b", "b"],
            "time_s": [0.0, 2.0, 0.0, 1.0],
            "truth_time_s": [0.0, 2.0, 0.0, 1.0],
            "matched": [True, True, True, True],
            "error_3d_m": [1.0, 20.0, 2.0, 100.0],
            "error_2d_m": [1.0, 12.0, 2.0, 60.0],
        }
    )
    submission = matches[["sequence_id", "track_id", "time_s"]].copy()
    truth = matches[["sequence_id", "track_id", "truth_time_s"]].rename(
        columns={"truth_time_s": "time_s"}
    )

    metrics = metrics_from_matches(matches, submission=submission, truth=truth)

    assert metrics["pooled"]["fde_3d_m"] == 60.0
    assert metrics["pooled"]["fde_2d_m"] == 36.0
    assert metrics["sequences"]["seq1"]["fde_3d_m"] == 60.0
    assert metrics["sequences"]["seq1"]["fde_2d_m"] == 36.0
