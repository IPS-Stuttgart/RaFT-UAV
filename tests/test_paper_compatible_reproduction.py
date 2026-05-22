from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.radar_association import _select_paper_compatible_radar_track
from raft_uav.diagnostics.paper_table import select_radar_for_table
from raft_uav.evaluation.metrics import summarize_errors


def test_summarize_errors_includes_paper_style_fields() -> None:
    summary = summarize_errors(np.array([3.0, 4.0]))

    assert summary["count"] == 2.0
    assert summary["mean_m"] == 3.5
    assert summary["std_m"] == 0.5
    assert summary["max_m"] == 4.0
    assert np.isclose(summary["rmse_m"], np.sqrt(12.5))


def test_summarize_errors_drops_nonfinite_values() -> None:
    summary = summarize_errors(np.array([np.nan, np.inf]))

    assert summary["count"] == 0.0
    assert summary["mean_m"] is None
    assert summary["std_m"] is None
    assert summary["max_m"] is None


def test_paper_compatible_preselector_is_range_and_catprob_gated() -> None:
    radar = pd.DataFrame(
        [
            *_rows(track_id=1, frames=range(4), range_m=900.0, catprob=0.95),
            *_rows(track_id=2, frames=range(3), range_m=700.0, catprob=0.80),
            *_rows(track_id=3, frames=range(5), range_m=650.0, catprob=0.10),
        ]
    )

    selected = _select_paper_compatible_radar_track(
        radar,
        range_gate_m=800.0,
        catprob_threshold=0.4,
    )

    assert selected["track_id"].astype(int).unique().tolist() == [2]
    assert selected["frame_index"].astype(int).tolist() == [0, 1, 2]
    assert selected["association_preselector_raw_rows"].max() == len(radar)
    assert selected["association_preselector_range_gated_rows"].max() == 8
    assert selected["association_preselector_catprob_rows"].max() == 3


def test_longest_continuous_table_selection_returns_one_segment_not_whole_id() -> None:
    radar = pd.DataFrame(
        [
            *_rows(track_id=8, frames=[0, 1, 2, 3], range_m=500.0, catprob=0.8),
            *_rows(track_id=8, frames=[10, 11], range_m=500.0, catprob=0.8),
            *_rows(track_id=9, frames=[4, 5, 6], range_m=500.0, catprob=0.9),
        ]
    )

    selected = select_radar_for_table(
        radar=radar,
        truth=pd.DataFrame(),
        selection="radar-longest-continuous-track-range-gated",
        catprob_threshold=0.4,
        range_gate_m=800.0,
        max_time_delta_s=float("inf"),
    )

    assert selected["track_id"].astype(int).unique().tolist() == [8]
    assert selected["frame_index"].astype(int).tolist() == [0, 1, 2, 3]
    assert selected["association_segment_frames"].max() == 4


def _rows(
    *,
    track_id: int,
    frames: range | list[int],
    range_m: float,
    catprob: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for frame in frames:
        rows.append(
            {
                "time_s": float(frame),
                "frame_index": int(frame),
                "track_id": int(track_id),
                "track_index": 0,
                "east_m": float(track_id * 10 + frame),
                "north_m": float(frame),
                "up_m": 10.0,
                "range_m": float(range_m),
                "cat_prob_uav": float(catprob),
            }
        )
    return rows
