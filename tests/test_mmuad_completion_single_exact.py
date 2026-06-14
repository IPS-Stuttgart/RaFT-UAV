import pandas as pd

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps


def test_single_exact_prediction_is_kept_without_hold_extrapolation():
    results = pd.DataFrame(
        {
            "sequence_id": ["seq1"],
            "timestamp": [2.0],
            "x": [1.0],
            "y": [2.0],
            "z": [3.0],
            "uav_type": ["2"],
            "score": [0.75],
        }
    )
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [2.0]})

    completed = complete_results_to_truth_timestamps(
        results,
        template,
        extrapolation="nan",
    )

    assert completed.rows.to_dict("records") == [
        {
            "sequence_id": "seq1",
            "timestamp": 2.0,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "uav_type": "2",
            "score": 0.75,
        }
    ]
    assert completed.diagnostics["completion_method"].tolist() == ["exact"]
