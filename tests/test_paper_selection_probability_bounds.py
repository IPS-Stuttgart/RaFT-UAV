from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.paper_selection import (
    _finite_catprob_value,
    _mean_catprob,
    range_gated_radar_candidates,
)


@pytest.mark.parametrize(
    "value",
    [-0.01, 1.01, np.nan, np.inf, -np.inf, True, 1.0 + 1.0j],
)
def test_probability_parser_rejects_invalid_values(value: object) -> None:
    assert _finite_catprob_value(value) is None


def test_probability_parser_accepts_closed_unit_interval() -> None:
    assert _finite_catprob_value(0.0) == 0.0
    assert _finite_catprob_value(np.float32(0.25)) == pytest.approx(0.25)
    assert _finite_catprob_value(1.0) == 1.0


def test_out_of_range_catprob_is_excluded_from_paper_gate() -> None:
    radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "track_id": [1, 1, 1],
            "range_m": [100.0, 100.0, 100.0],
            "cat_prob_uav": [-0.2, 1.2, 0.8],
        }
    )

    selected = range_gated_radar_candidates(
        radar,
        catprob_threshold=0.5,
        require_range_m=False,
    )

    assert selected["time_s"].tolist() == [2.0]
    assert selected["association_catprob_candidate_rows"].tolist() == [3]


def test_track_tie_break_mean_ignores_out_of_range_probabilities() -> None:
    radar = pd.DataFrame(
        {
            "cat_prob_uav": [0.9, 2.0, -0.1, np.nan],
        }
    )

    assert _mean_catprob(radar) == pytest.approx(0.9)
