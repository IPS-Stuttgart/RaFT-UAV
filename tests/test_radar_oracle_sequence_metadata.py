from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.evaluation.radar_oracle_diagnostics import nearest_candidate_oracle


def _radar(*, sequences: list[object] | None) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "frame_index": [0, 0],
        "track_id": [11, 22],
        "time_s": [0.0, 0.0],
        "east_m": [0.0, 100.0],
        "north_m": [0.0, 0.0],
        "up_m": [0.0, 0.0],
    }
    if sequences is not None:
        data["sequence_id"] = sequences
    return pd.DataFrame(data)


def _truth(*, sequences: list[object] | None) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "time_s": [0.0, 0.0],
        "east_m": [0.0, 100.0],
        "north_m": [0.0, 0.0],
        "up_m": [0.0, 0.0],
    }
    if sequences is not None:
        data["sequence_id"] = sequences
    return pd.DataFrame(data)


def test_oracle_rejects_pooled_radar_against_unlabeled_truth() -> None:
    radar = _radar(sequences=["flight-a", "flight-b"])
    truth = _truth(sequences=None)

    with pytest.raises(ValueError, match="pooled radar.*require truth sequence_id"):
        nearest_candidate_oracle(radar, truth)


def test_oracle_rejects_unlabeled_radar_against_pooled_truth() -> None:
    radar = _radar(sequences=None)
    truth = _truth(sequences=["flight-a", "flight-b"])

    with pytest.raises(ValueError, match="pooled truth.*require radar sequence_id"):
        nearest_candidate_oracle(radar, truth)


@pytest.mark.parametrize(
    ("radar_sequences", "truth_sequences", "message"),
    [
        (["flight-a", None], ["flight-a", "flight-a"], "radar sequence_id is partially"),
        (["flight-a", "flight-a"], ["flight-a", None], "truth sequence_id is partially"),
    ],
)
def test_oracle_rejects_partially_labeled_sequence_metadata(
    radar_sequences: list[object],
    truth_sequences: list[object],
    message: str,
) -> None:
    radar = _radar(sequences=radar_sequences)
    truth = _truth(sequences=truth_sequences)

    with pytest.raises(ValueError, match=message):
        nearest_candidate_oracle(radar, truth)


def test_oracle_preserves_one_sided_single_sequence_compatibility() -> None:
    radar = _radar(sequences=["flight-a", "flight-a"])
    truth = _truth(sequences=[None, None])

    selected = nearest_candidate_oracle(radar, truth, max_time_delta_s=0.1)

    assert len(selected) == 1
    assert selected.loc[0, "sequence_id"] == "flight-a"
    assert selected.loc[0, "oracle_error_3d_m"] == 0.0
