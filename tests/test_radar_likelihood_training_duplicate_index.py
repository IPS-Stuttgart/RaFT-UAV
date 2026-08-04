import pandas as pd

from raft_uav.baselines import radar_likelihood_training as training


def test_prediction_nis_selection_is_positional_with_duplicate_index():
    scored = pd.DataFrame(
        {
            "association_score": [3.0, 1.0, 2.0],
            "association_nis": [3.0, 1.0, 2.0],
            "track_id": [1, 2, 3],
        },
        index=[7, 7, 8],
    )

    selected = training._student_selected_candidate(
        scored,
        teacher_association="prediction-nis",
        current_track_id=None,
        track_switch_nis_ratio=0.5,
    )

    assert isinstance(selected, pd.Series)
    assert selected["association_score"] == 1.0
    assert selected["track_id"] == 2


def test_track_continuity_selection_is_positional_with_duplicate_current_index():
    scored = pd.DataFrame(
        {
            "association_score": [2.0, 4.0, 1.0],
            "association_nis": [2.0, 4.0, 3.0],
            "track_id": [5, 5, 9],
        },
        index=[7, 7, 8],
    )

    selected = training._student_selected_candidate(
        scored,
        teacher_association="track-continuity",
        current_track_id=5,
        track_switch_nis_ratio=0.5,
    )

    assert isinstance(selected, pd.Series)
    assert selected["association_score"] == 2.0
    assert selected["track_id"] == 5
