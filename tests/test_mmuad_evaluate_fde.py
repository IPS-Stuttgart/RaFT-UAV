import pandas as pd

from raft_uav.mmuad.evaluate import metrics_from_matches


def test_submission_eval_fde_uses_latest_time_not_input_order():
    matches = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "time_s": [2.0, 0.0, 1.0],
            "truth_time_s": [2.0, 0.0, 1.0],
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
