from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.diagnostics.paper_parity_grid import (
    PaperParityGridCandidate,
    _parse_grid,
    _parse_nullable_float_list,
    build_candidate_grid,
    rank_grid_summary,
)


def test_parse_grid_is_inclusive() -> None:
    grid = _parse_grid("-0.5,0.5,0.5")

    assert np.allclose(grid, np.array([-0.5, 0.0, 0.5]))


def test_parse_grid_rejects_negative_step() -> None:
    with pytest.raises(ValueError, match="STEP"):
        _parse_grid("0,1,-0.1")


def test_parse_nullable_float_list_accepts_none_and_commas() -> None:
    values = _parse_nullable_float_list(["none,0.4", "0.5"])

    assert values == (None, 0.4, 0.5)


def test_build_candidate_grid_returns_cartesian_product() -> None:
    candidates = build_candidate_grid(
        flights=["Opt1"],
        variants=["original", "rerun"],
        radar_track_selection_orders=["raw-track-then-range"],
        bootstrap_sources=["radar", "first-event"],
        radar_catprob_thresholds=[None, 0.4],
        rf_residual_grid_s=np.array([0.0]),
        radar_residual_grid_s=np.array([-0.25, 0.0]),
    )

    assert len(candidates) == 16
    assert candidates[0] == PaperParityGridCandidate(
        flight="Opt1",
        variant="original",
        radar_track_selection_order="raw-track-then-range",
        bootstrap_source="radar",
        radar_catprob_threshold=None,
        rf_residual_offset_s=0.0,
        radar_residual_offset_s=-0.25,
    )


def test_rank_grid_summary_sorts_failures_after_successes() -> None:
    summary = pd.DataFrame(
        [
            {
                "variant": "rerun",
                "radar_track_selection_order": "raw-track-then-range",
                "bootstrap_source": "radar",
                "failed": True,
                "paper_parity_score": 0.0,
                "count_abs_delta_total": 0,
                "kf_all_steps_mean_abs_delta_m": 0.0,
            },
            {
                "variant": "original",
                "radar_track_selection_order": "raw-track-then-range",
                "bootstrap_source": "radar",
                "failed": False,
                "paper_parity_score": 10.0,
                "count_abs_delta_total": 0,
                "kf_all_steps_mean_abs_delta_m": 10.0,
            },
            {
                "variant": "rerun",
                "radar_track_selection_order": "raw-track-then-range",
                "bootstrap_source": "first-event",
                "failed": False,
                "paper_parity_score": 5.0,
                "count_abs_delta_total": 0,
                "kf_all_steps_mean_abs_delta_m": 5.0,
            },
        ]
    )

    ranked = rank_grid_summary(summary)

    assert ranked["rank"].tolist() == [1, 2, 3]
    assert ranked["bootstrap_source"].tolist() == ["first-event", "radar", "radar"]
    assert ranked["failed"].tolist() == [False, False, True]
