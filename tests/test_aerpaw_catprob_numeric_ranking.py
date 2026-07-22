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


def test_catprob_best_per_frame_falls_back_from_incomplete_frame_index() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [0, pd.NA, 1, pd.NA],
            "time_s": [10.0, 10.0, 11.0, 11.0],
            "cat_prob_uav": [0.8, 0.9, 0.7, 0.6],
            "track_id": [1, 2, 3, 4],
        }
    )

    selected = _catprob_best_per_frame_rows(radar, catprob_threshold=0.0)

    assert selected["track_id"].tolist() == [2, 3]


def test_catprob_best_per_frame_ignores_all_missing_frame_index() -> None:
    radar = pd.DataFrame(
        {
            "frame_index": [pd.NA, pd.NA, pd.NA, pd.NA],
            "time_s": [10.0, 10.0, 11.0, 11.0],
            "cat_prob_uav": [0.8, 0.9, 0.7, 0.6],
            "track_id": [1, 2, 3, 4],
        }
    )

    selected = _catprob_best_per_frame_rows(radar, catprob_threshold=0.0)

    assert selected["track_id"].tolist() == [2, 3]
