import pandas as pd

from raft_uav.mmuad.completion import complete_results_to_truth_timestamps
from raft_uav.mmuad.evaluator import validate_mmaud_results_frame


def _result_rows(sequence_ids):
    count = len(sequence_ids)
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "timestamp": list(range(count)),
            "x": [1.0] * count,
            "y": [2.0] * count,
            "z": [3.0] * count,
            "uav_type": ["2"] * count,
            "score": [0.75] * count,
        }
    )


def test_local_result_sequence_ids_are_stripped_and_missing_values_use_default():
    validated = validate_mmaud_results_frame(
        _result_rows([" flight-1 ", None, "", "NaN", "<NA>", "NaT"])
    )

    assert validated["sequence_id"].tolist() == [
        "default",
        "default",
        "default",
        "default",
        "default",
        "flight-1",
    ]


def test_completion_matches_padded_local_result_sequence_id_to_template():
    completed = complete_results_to_truth_timestamps(
        _result_rows([" flight-1 "]),
        pd.DataFrame({"sequence_id": ["flight-1"], "time_s": [0.0]}),
        extrapolation="nan",
    )

    assert completed.rows["sequence_id"].tolist() == ["flight-1"]
    assert completed.diagnostics["completion_method"].tolist() == ["exact"]
