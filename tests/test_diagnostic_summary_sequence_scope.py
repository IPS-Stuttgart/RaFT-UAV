from __future__ import annotations

import math

import pandas as pd
import pytest

from raft_uav.evaluation.diagnostics import build_diagnostic_summary


def _estimate_rows(sequence_ids: list[str]) -> pd.DataFrame:
    count = len(sequence_ids)
    return pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "time_s": [0.0] * count,
            "source": ["rf"] * count,
            "east_m": [0.0, 100.0][:count],
            "north_m": [0.0] * count,
            "up_m": [0.0] * count,
            "residual_norm_m": [float("nan")] * count,
            "covariance_scale": [1.0] * count,
        }
    )


def test_worst_windows_align_pooled_runs_within_sequence() -> None:
    estimates = _estimate_rows(["seq_a", "seq_b"])
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq_a", "seq_b"],
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    summary = build_diagnostic_summary(
        estimate_frame=estimates,
        selected_radar=pd.DataFrame(),
        truth=truth,
        max_eval_time_delta_s=0.0,
        window_s=10.0,
    )

    assert summary["worst_time_windows"][0]["count"] == 2
    assert math.isclose(summary["worst_time_windows"][0]["rmse_3d_m"], 0.0)


def test_pooled_one_sided_sequence_metadata_fails_closed() -> None:
    estimates = _estimate_rows(["seq_a", "seq_b"])
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 0.0],
            "east_m": [0.0, 100.0],
            "north_m": [0.0, 0.0],
            "up_m": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="without matching sequence metadata"):
        build_diagnostic_summary(
            estimate_frame=estimates,
            selected_radar=pd.DataFrame(),
            truth=truth,
            max_eval_time_delta_s=0.0,
        )


def test_single_sequence_one_sided_metadata_remains_compatible() -> None:
    estimates = _estimate_rows(["seq_a"]).assign(east_m=[5.0])
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [2.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    summary = build_diagnostic_summary(
        estimate_frame=estimates,
        selected_radar=pd.DataFrame(),
        truth=truth,
        max_eval_time_delta_s=0.0,
    )

    assert math.isclose(summary["worst_time_windows"][0]["rmse_3d_m"], 3.0)


def test_conflicting_sequence_aliases_are_rejected() -> None:
    estimates = _estimate_rows(["seq_a"]).assign(flight_id=["seq_b"])
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq_a"],
            "time_s": [0.0],
            "east_m": [0.0],
            "north_m": [0.0],
            "up_m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="conflicting sequence_id and flight_id"):
        build_diagnostic_summary(
            estimate_frame=estimates,
            selected_radar=pd.DataFrame(),
            truth=truth,
            max_eval_time_delta_s=0.0,
        )


def test_track_switches_are_scoped_to_each_pooled_sequence() -> None:
    selected_radar = pd.DataFrame(
        {
            "sequence_id": ["seq_a", "seq_b", "seq_a", "seq_b"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "track_id": [10, 20, 10, 20],
        }
    )

    summary = build_diagnostic_summary(
        estimate_frame=pd.DataFrame(),
        selected_radar=selected_radar,
        truth=pd.DataFrame(),
        max_eval_time_delta_s=None,
    )

    switches = summary["track_switches"]["selected_radar"]
    assert switches["count"] == 0
    assert switches["updates_with_track_id"] == 4
    assert switches["unique_track_ids"] == 2
    assert switches["top_transitions"] == []
    assert switches["events"] == []


def test_track_switches_preserve_input_order_for_equal_timestamps() -> None:
    time_s = [2, 1, 1, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1, 2]
    selected_radar = pd.DataFrame(
        {
            "time_s": time_s,
            "track_id": [1] * 10 + [2] * 10,
        }
    )

    summary = build_diagnostic_summary(
        estimate_frame=pd.DataFrame(),
        selected_radar=selected_radar,
        truth=pd.DataFrame(),
        max_eval_time_delta_s=None,
    )

    switches = summary["track_switches"]["selected_radar"]
    assert switches["count"] == 3
    assert switches["events"] == [
        {"from_track_id": 1, "to_track_id": 2, "time_s": 1},
        {"from_track_id": 2, "to_track_id": 1, "time_s": 2},
        {"from_track_id": 1, "to_track_id": 2, "time_s": 2},
    ]
