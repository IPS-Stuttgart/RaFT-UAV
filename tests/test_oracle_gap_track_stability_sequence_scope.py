from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.evaluation.oracle_gap_decomposition import (
    selected_track_stability_metrics,
)


def test_track_stability_excludes_cross_sequence_transitions_and_gaps() -> None:
    selected = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 100.0, 101.0],
            "track_id": [1, 1, 2, 2],
        }
    )

    metrics = selected_track_stability_metrics(selected)

    assert metrics["selected_sequence_count"] == 2
    assert metrics["track_switch_count"] == 0
    assert metrics["track_switch_rate"] == 0.0
    assert metrics["selected_time_gap_p95_s"] == 1.0
    assert metrics["selected_time_gap_max_s"] == 1.0


def test_track_stability_sums_only_within_sequence_transitions() -> None:
    selected = pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB", "seqB"],
            "time_s": [0.0, 1.0, 100.0, 101.0],
            "track_id": [1, 2, 3, 3],
        }
    )

    metrics = selected_track_stability_metrics(selected)

    assert metrics["track_switch_count"] == 1
    assert metrics["track_switch_rate"] == pytest.approx(0.5)
