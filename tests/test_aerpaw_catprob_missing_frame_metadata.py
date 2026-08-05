import pandas as pd

from raft_uav.io.aerpaw import _catprob_best_per_frame_rows


def test_catprob_best_per_frame_preserves_rows_without_frame_metadata() -> None:
    radar = pd.DataFrame(
        {
            "cat_prob_uav": [0.8, 0.9, 0.7],
            "track_id": ["first", "second", "third"],
        }
    )

    selected = _catprob_best_per_frame_rows(radar, catprob_threshold=0.0)

    assert selected["track_id"].tolist() == ["first", "second", "third"]
    assert selected["cat_prob_uav"].tolist() == [0.8, 0.9, 0.7]
