from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines import radar_association as radar_association_module
from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.radar_association import (
    _run_paper_compatible_association,
    _select_paper_compatible_radar_track,
)
from raft_uav.diagnostics.paper_table import select_radar_for_table
from raft_uav.evaluation.metrics import summarize_errors
from raft_uav.paper_selection import (
    paper_radar_track_stages,
    select_paper_compatible_radar_track as shared_select_paper_compatible_radar_track,
)


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


def test_paper_compatible_empirical_rf_covariance_rebuilds_events(monkeypatch) -> None:
    original_covariance = np.diag([1.0, 1.0])
    empirical_covariance = np.diag([49.0, 64.0])
    rf_measurements = [
        TrackingMeasurement(
            time_s=0.0,
            vector=np.array([10.0, 20.0]),
            covariance=original_covariance,
            source="rf",
            _apply_runtime_calibration=False,
        )
    ]
    radar = pd.DataFrame(columns=["time_s", "east_m", "north_m", "up_m"])
    truth = pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [10.0],
            "north_m": [20.0],
            "up_m": [0.0],
        }
    )
    observed_covariances: list[np.ndarray] = []

    def fake_events(rf_measurements_arg, _radar):
        measurement = rf_measurements_arg[0]
        observed_covariances.append(measurement.covariance.copy())
        return [
            {
                "time_s": measurement.time_s,
                "priority": 0,
                "kind": "rf",
                "measurement": measurement,
            }
        ]

    def fake_empirical_rf_covariance(_measurements, **_kwargs):
        return empirical_covariance

    def fake_initial(events, **_kwargs):
        np.testing.assert_allclose(
            events[0]["measurement"].covariance,
            empirical_covariance,
        )
        return None

    monkeypatch.setattr(radar_association_module, "_events", fake_events)
    monkeypatch.setattr(
        radar_association_module,
        "_empirical_rf_covariance_from_measurements",
        fake_empirical_rf_covariance,
    )
    monkeypatch.setattr(
        radar_association_module,
        "_initial_paper_compatible_measurement_and_row",
        fake_initial,
    )

    records, selected = _run_paper_compatible_association(
        rf_measurements=rf_measurements,
        radar=radar,
        covariance=np.eye(3),
        covariance_config=None,
        acceleration_std_mps2=1.0,
        gate_probabilities_by_source=None,
        gate_thresholds_by_source=None,
        safety_gate_probabilities_by_source=None,
        safety_gate_thresholds_by_source=None,
        robust_update_by_source=None,
        inflation_alpha_by_source=None,
        max_residual_norms_by_source=None,
        candidate_catprob_threshold=None,
        range_gate_m=None,
        track_switch_penalty=0.0,
        catprob_weight=0.0,
        bootstrap_source="first-event",
        truth=truth,
        empirical_covariance=True,
        truth_time_gate_s=1.0,
    )

    assert records == []
    assert selected.empty
    assert len(observed_covariances) == 2
    np.testing.assert_allclose(observed_covariances[0], original_covariance)
    np.testing.assert_allclose(observed_covariances[1], empirical_covariance)


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


def test_paper_compatible_preselector_uses_shared_selection() -> None:
    radar = pd.DataFrame(
        [
            *_rows(track_id=1, frames=range(4), range_m=900.0, catprob=0.95),
            *_rows(track_id=2, frames=range(3), range_m=700.0, catprob=0.80),
            *_rows(track_id=3, frames=range(5), range_m=650.0, catprob=0.10),
        ]
    )

    online = _select_paper_compatible_radar_track(
        radar,
        range_gate_m=800.0,
        catprob_threshold=0.4,
    )
    shared = shared_select_paper_compatible_radar_track(
        radar,
        range_gate_m=800.0,
        catprob_threshold=0.4,
    )

    compare_columns = [
        "frame_index",
        "track_id",
        "association_preselector_raw_rows",
        "association_preselector_range_gated_rows",
        "association_preselector_catprob_rows",
        "association_preselector_track_rows",
    ]
    pd.testing.assert_frame_equal(
        online[compare_columns].reset_index(drop=True),
        shared[compare_columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_paper_radar_track_stages_make_selection_order_explicit() -> None:
    radar = pd.DataFrame(
        [
            *_rows(track_id=1, frames=range(4), range_m=900.0, catprob=0.95),
            *_rows(track_id=2, frames=range(3), range_m=700.0, catprob=0.80),
        ]
    )

    raw_first = paper_radar_track_stages(
        radar,
        range_gate_m=800.0,
        radar_track_selection_order="raw-track-then-range",
    )
    range_first = paper_radar_track_stages(
        radar,
        range_gate_m=800.0,
        radar_track_selection_order="range-then-largest-track",
    )

    assert raw_first.raw_target["track_id"].astype(int).unique().tolist() == [1]
    assert raw_first.preselected.empty
    assert range_first.raw_target["track_id"].astype(int).unique().tolist() == [2]
    assert range_first.preselected["frame_index"].astype(int).tolist() == [0, 1, 2]


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
