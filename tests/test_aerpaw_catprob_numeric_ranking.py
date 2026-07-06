import pandas as pd

from raft_uav.io.aerpaw import _catprob_best_per_frame_rows, _catprob_threshold_rows


def test_catprob_threshold_rows_normalizes_text_probabilities() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 0],
            "cat_prob_uav": ["1e-1", "0.9"],
            "track_id": ["low", "high"],
        }
    )

    selected = _catprob_threshold_rows(radar, catprob_threshold=0.0)

    assert selected["cat_prob_uav"].tolist() == [0.1, 0.9]


def test_catprob_best_per_frame_ranks_numeric_text_probabilities() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, 0],
            "cat_prob_uav": ["1e-1", "0.9"],
            "track_id": ["low", "high"],
        }
    )

    selected = _catprob_best_per_frame_rows(radar, catprob_threshold=0.0)

    assert selected["track_id"].tolist() == ["high"]
    assert selected["cat_prob_uav"].tolist() == [0.9]
