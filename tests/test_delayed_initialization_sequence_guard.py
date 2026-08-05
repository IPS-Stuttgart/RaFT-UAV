from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.baselines.delayed_initialization import (
    build_delayed_initial_hypotheses,
)


def _radar(sequence_ids: list[object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "time_s": [0.0, 1.0, 0.0, 1.0][: len(sequence_ids)],
            "track_id": [1] * len(sequence_ids),
            "east_m": [0.0, 1.0, 100.0, 101.0][: len(sequence_ids)],
            "north_m": [0.0] * len(sequence_ids),
            "up_m": [0.0] * len(sequence_ids),
            "cat_prob_uav": [1.0] * len(sequence_ids),
        }
    )


def test_delayed_initialization_rejects_pooled_radar_sequences() -> None:
    radar = _radar(["flight-a", "flight-a", "flight-b", "flight-b"])

    with pytest.raises(
        ValueError,
        match="delayed initialization requires radar from one sequence_id",
    ):
        build_delayed_initial_hypotheses(
            rf_measurements=[],
            radar=radar,
        )


def test_delayed_initialization_allows_one_explicit_sequence_with_missing_labels() -> None:
    radar = _radar([" flight-a ", None]).iloc[:2].copy()

    hypotheses = build_delayed_initial_hypotheses(
        rf_measurements=[],
        radar=radar,
    )

    assert len(hypotheses) == 2
    for hypothesis in hypotheses:
        assert hypothesis.state[3:6] == pytest.approx([1.0, 0.0, 0.0])
        assert hypothesis.metadata["support_score"] == pytest.approx(0.5)
