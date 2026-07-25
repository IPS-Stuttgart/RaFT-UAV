import pandas as pd

from raft_uav.mmuad.candidate_pull import align_candidate_centers


def test_candidate_center_alignment_preserves_interleaved_result_order():
    results = pd.DataFrame(
        {
            "Sequence": ["B", "A", "B", "A"],
            "Timestamp": [2.0, 1.0, 1.0, 2.0],
            "current_x": [20.0, 10.0, 11.0, 12.0],
        }
    )
    centers = pd.DataFrame(
        {
            "Sequence": ["A", "A", "B", "B"],
            "candidate_time_s": [1.0, 2.0, 1.0, 2.0],
            "topk_dispersion_m": [101.0, 102.0, 201.0, 202.0],
        }
    )

    aligned = align_candidate_centers(
        results,
        centers,
        time_tolerance_s=0.01,
    )

    assert aligned["Sequence"].tolist() == results["Sequence"].tolist()
    assert aligned["Timestamp"].tolist() == results["Timestamp"].tolist()
    assert aligned["current_x"].tolist() == results["current_x"].tolist()
    assert aligned["topk_dispersion_m"].tolist() == [202.0, 101.0, 201.0, 102.0]
