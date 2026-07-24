from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.oracle_gap_decomposition import (
    OracleGapConfig,
    decompose_radar_oracle_gap,
    selected_track_stability_metrics,
)


def test_track_stability_ignores_malformed_track_identifiers() -> None:
    selected = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "track_id": pd.Series(
                [12.75, 12.75, True, "9", 9.0, "9.0"],
                dtype=object,
            ),
        }
    )

    metrics = selected_track_stability_metrics(selected)

    assert metrics["selected_radar_rows"] == 6
    assert metrics["finite_track_id_rows"] == 3
    assert metrics["unique_selected_track_ids"] == 1
    assert metrics["track_switch_count"] == 0
    assert metrics["dominant_track_id"] == 9
    assert metrics["dominant_track_fraction"] == pytest.approx(1.0)
    assert metrics["selected_track_entropy"] == pytest.approx(0.0)


@pytest.mark.parametrize("invalid_track_id", [4.5, True, np.bool_(False)])
def test_oracle_gap_does_not_truncate_invalid_track_identifiers(
    invalid_track_id: object,
) -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_index": [0],
            "track_id": pd.Series([invalid_track_id], dtype=object),
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        selected_radar=radar,
        config=OracleGapConfig(plausible_candidate_gate_m=10.0),
    )

    assert rows.loc[0, "nearest_candidate_track_id"] == ""
    assert rows.loc[0, "selected_track_id"] == ""


def test_oracle_gap_keeps_integer_equivalent_track_identifiers() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0],
            "frame_index": [0],
            "track_id": ["9.0"],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    rows = decompose_radar_oracle_gap(
        radar=radar,
        truth=truth,
        selected_radar=radar,
        config=OracleGapConfig(plausible_candidate_gate_m=10.0),
    )

    assert rows.loc[0, "nearest_candidate_track_id"] == 9
    assert rows.loc[0, "selected_track_id"] == 9
