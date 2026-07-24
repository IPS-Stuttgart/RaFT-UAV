from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.mot import compute_multi_object_metrics


def _estimates(*, include_sequence_id: bool) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "time_s": [0.0],
        "state_x_m": [100.0],
        "state_y_m": [0.0],
        "state_z_m": [0.0],
        "output_track_id": ["prediction"],
    }
    if include_sequence_id:
        data["sequence_id"] = ["seq_b"]
    return pd.DataFrame(data)


def _truth(*, include_sequence_id: bool) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "time_s": [0.0, 0.0],
        "x_m": [0.0, 100.0],
        "y_m": [0.0, 0.0],
        "z_m": [0.0, 0.0],
        "track_id": ["object_a", "object_b"],
    }
    if include_sequence_id:
        data["sequence_id"] = ["seq_a", "seq_b"]
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("estimates_have_sequence_id", "truth_has_sequence_id"),
    [(True, False), (False, True)],
)
def test_mot_metrics_reject_one_sided_sequence_metadata(
    estimates_have_sequence_id: bool,
    truth_has_sequence_id: bool,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "estimates and truth must either both contain sequence_id or both omit it"
        ),
    ):
        compute_multi_object_metrics(
            _estimates(include_sequence_id=estimates_have_sequence_id),
            _truth(include_sequence_id=truth_has_sequence_id),
            match_distance_m=1.0,
        )


def test_mot_metrics_allow_truth_only_sequence_metadata() -> None:
    metrics = compute_multi_object_metrics(
        pd.DataFrame(),
        _truth(include_sequence_id=True),
        match_distance_m=1.0,
    )

    assert metrics["matches"] == 0
    assert metrics["false_negative"] == 2
