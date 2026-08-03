from __future__ import annotations

import pandas as pd

from raft_uav.paper_selection import (
    _finite_catprob_value,
    range_gated_radar_candidates,
    select_paper_strict_raw_radar_track,
)


def test_probability_parser_enforces_closed_unit_interval() -> None:
    assert _finite_catprob_value(-0.01) is None
    assert _finite_catprob_value(1.01) is None
    assert _finite_catprob_value(0.0) == 0.0
    assert _finite_catprob_value(1.0) == 1.0


def test_above_one_catprob_is_excluded_from_paper_gate() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "track_id": [1, 1],
            "range_m": [100.0, 100.0],
            "cat_prob_uav": [1.2, 0.8],
        }
    )

    selected = range_gated_radar_candidates(
        radar,
        catprob_threshold=0.5,
        require_range_m=False,
    )

    assert selected["time_s"].tolist() == [1.0]
    assert selected["association_catprob_candidate_rows"].tolist() == [2]


def test_above_one_catprob_does_not_win_track_tie_break() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "track_id": [1, 1, 2, 2],
            "cat_prob_uav": [0.9, 0.9, 2.0, 0.1],
        }
    )

    selected = select_paper_strict_raw_radar_track(radar)

    assert selected["track_id"].unique().tolist() == [1]
