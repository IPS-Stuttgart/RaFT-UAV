from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps


def test_duplicate_timestamp_completion_is_order_independent_and_prefers_highest_score():
    results = pd.DataFrame(
        {
            "sequence_id": ["seq1", "seq1", "seq1"],
            "timestamp": [2.0, 2.0, 2.0],
            "x": [4.0, 1.0, 0.0],
            "y": [5.0, 2.0, 0.0],
            "z": [6.0, 3.0, 0.0],
            "uav_type": ["2", "2", "2"],
            "score": [0.9, 0.9, 0.2],
        }
    )
    template = pd.DataFrame({"sequence_id": ["seq1"], "time_s": [2.0]})

    forward = complete_results_to_truth_timestamps(results, template, extrapolation="nan")
    reverse = complete_results_to_truth_timestamps(
        results.iloc[::-1].reset_index(drop=True),
        template,
        extrapolation="nan",
    )

    expected = [
        {
            "sequence_id": "seq1",
            "timestamp": 2.0,
            "x": 1.0,
            "y": 2.0,
            "z": 3.0,
            "uav_type": "2",
            "score": 0.9,
        }
    ]
    assert forward.rows.to_dict("records") == expected
    assert reverse.rows.to_dict("records") == expected
    assert forward.diagnostics["completion_method"].tolist() == ["exact"]
    assert reverse.diagnostics["completion_method"].tolist() == ["exact"]
