from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.kalman import TrackingMeasurement
from raft_uav.baselines.topk_weakz_tracklet import (
    TopKWeakZTrackletConfig,
    build_fortem_tracklets,
    radar_tracklet_measurements,
    rf_measurements_with_radar_reliability,
    run_topk_tracklet_graph_weakz_smoother,
    selected_radar_for_tracklet_path,
    top_k_tracklet_paths,
)


def _radar_frame() -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append(
            {
                "frame_index": i,
                "track_id": 10,
                "time_s": float(i),
                "east_m": float(10.0 * i),
                "north_m": 0.0,
                "up_m": 100.0,
                "range_m": 450.0,
                "cat_prob_uav": 0.95,
                "confidence": 0.8,
            }
        )
    for i in range(8):
        rows.append(
            {
                "frame_index": i,
                "track_id": 20,
                "time_s": float(i),
                "east_m": float(500.0 + 20.0 * i),
                "north_m": 500.0,
                "up_m": 40.0,
                "range_m": 700.0,
                "cat_prob_uav": 0.15,
                "confidence": 0.2,
            }
        )
    return pd.DataFrame.from_records(rows)


def _rf_measurements() -> list[TrackingMeasurement]:
    return [
        TrackingMeasurement(
            time_s=float(i),
            vector=np.array([10.0 * i, 1.0]),
            covariance=np.diag([50.0**2, 50.0**2]),
            source="rf",
        )
        for i in range(8)
    ]


def test_build_fortem_tracklets_splits_by_track_id() -> None:
    cfg = TopKWeakZTrackletConfig(min_tracklet_length=3)
    tracklets = build_fortem_tracklets(_radar_frame(), cfg)
    assert len(tracklets) == 2
    assert {tracklet.track_id for tracklet in tracklets} == {10, 20}
    assert all(tracklet.row_count == 8 for tracklet in tracklets)
    best = min(tracklets, key=lambda tracklet: tracklet.unary_cost)
    assert best.track_id == 10


def test_selected_radar_for_tracklet_path_uses_positions_not_external_index() -> None:
    radar = _radar_frame()
    radar.index = np.arange(100, 100 + len(radar))
    cfg = TopKWeakZTrackletConfig(top_k_paths=2, beam_width=8, min_tracklet_length=3)
    tracklets = build_fortem_tracklets(radar, cfg)
    path = top_k_tracklet_paths(tracklets, cfg)[0]
    segment_by_id = {segment.segment_id: segment for segment in tracklets}

    selected = selected_radar_for_tracklet_path(radar, path, segment_by_id)

    assert not selected.empty
    assert selected["track_id"].eq(10).all()
    assert selected["east_m"].tolist() == [10.0 * i for i in range(8)]
    assert selected["topk_weakz_segment_id"].ge(0).all()
    assert selected["topk_weakz_path_order"].eq(0).all()


def test_top_k_paths_prefers_high_quality_tracklet() -> None:
    cfg = TopKWeakZTrackletConfig(top_k_paths=2, beam_width=8, min_tracklet_length=3)
    tracklets = build_fortem_tracklets(_radar_frame(), cfg)
    paths = top_k_tracklet_paths(tracklets, cfg)
    assert len(paths) == 2
    segment_by_id = {segment.segment_id: segment for segment in tracklets}
    best_track_ids = [segment_by_id[segment_id].track_id for segment_id in paths[0].segment_ids]
    assert best_track_ids == [10]


def test_weakz_radar_measurements_use_large_vertical_covariance() -> None:
    cfg = TopKWeakZTrackletConfig(weakz_radar_xy_std_m=360.0, weakz_radar_z_std_m=20000.0)
    tracklets = build_fortem_tracklets(_radar_frame(), cfg)
    path = top_k_tracklet_paths(tracklets, cfg)[0]
    selected = selected_radar_for_tracklet_path(
        _radar_frame(),
        path,
        {segment.segment_id: segment for segment in tracklets},
    )
    measurements = radar_tracklet_measurements(selected, cfg)
    assert measurements
    covariance = measurements[0].covariance
    assert covariance[0, 0] == 360.0**2
    assert covariance[2, 2] == 20000.0**2


def test_rf_reliability_inflates_far_rf_measurement() -> None:
    cfg = TopKWeakZTrackletConfig(
        rf_soft_weight=True,
        rf_radar_consistency_std_m=20.0,
        rf_max_covariance_scale=10.0,
    )
    selected_radar = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "east_m": [0.0, 10.0, 20.0],
            "north_m": [0.0, 0.0, 0.0],
        }
    )
    rf = [
        TrackingMeasurement(
            time_s=1.0,
            vector=np.array([1000.0, 1000.0]),
            covariance=np.diag([10.0**2, 10.0**2]),
            source="rf",
        )
    ]
    weighted = rf_measurements_with_radar_reliability(rf, selected_radar, cfg)
    assert len(weighted) == 1
    assert weighted[0].covariance[0, 0] > rf[0].covariance[0, 0]
    assert weighted[0].covariance[0, 0] <= rf[0].covariance[0, 0] * 10.0


def test_run_topk_tracklet_graph_weakz_smoother_returns_diagnostics() -> None:
    cfg = TopKWeakZTrackletConfig(
        top_k_paths=2,
        beam_width=8,
        min_tracklet_length=3,
        smoother="none",
        acceleration_std_mps2=4.0,
        weakz_radar_xy_std_m=100.0,
        weakz_radar_z_std_m=20000.0,
    )
    result = run_topk_tracklet_graph_weakz_smoother(
        rf_measurements=_rf_measurements(),
        radar=_radar_frame(),
        config=cfg,
    )
    assert result.records
    assert not result.selected_radar.empty
    assert not result.path_diagnostics.empty
    assert not result.tracklet_diagnostics.empty
    assert result.selected_path_summary["radar_rows"] == len(result.selected_radar)
    assert result.selected_path_summary["weakz_radar_z_std_m"] == 20000.0
