from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from raft_uav.multi_uav_lts.upstream_patch import _ASSOCIATION_HELPER


class _Track:
    def __init__(self, xywh: tuple[float, float, float, float]) -> None:
        self._xywh = np.asarray(xywh, dtype=float)
        self.tlwh = np.asarray(
            [xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[2], xywh[3]],
            dtype=float,
        )
        self.mean = np.zeros(8)
        self.covariance = np.eye(8)
        self.smooth_feat = np.asarray([1.0, 0.0])
        self.curr_feat = np.asarray([1.0, 0.0])
        self.score = 1.0

    def to_xywh(self) -> np.ndarray:
        return self._xywh.copy()


def _write_helper(root: Path) -> Path:
    path = root / "tracker" / "raft_uav_association.py"
    path.parent.mkdir(parents=True)
    path.write_text(_ASSOCIATION_HELPER, encoding="utf-8")
    return path


def _load_helper(path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    matching = ModuleType("tracker.matching")

    def iou_distance(tracks: list[_Track], detections: list[_Track]) -> np.ndarray:
        result = np.zeros((len(tracks), len(detections)), dtype=float)
        for row, track in enumerate(tracks):
            for column, detection in enumerate(detections):
                result[row, column] = min(
                    1.0,
                    abs(track.to_xywh()[0] - detection.to_xywh()[0]) / 10.0,
                )
        return result

    matching.iou_distance = iou_distance
    matching.fuse_score = lambda costs, detections: costs
    matching.embedding_distance = lambda tracks, detections: np.zeros(
        (len(tracks), len(detections)), dtype=float
    )

    kalman_filter = ModuleType("tracker.kalman_filter")
    kalman_filter.chi2inv95 = {4: 9.4877}
    tracker = ModuleType("tracker")
    tracker.matching = matching
    tracker.kalman_filter = kalman_filter
    monkeypatch.setitem(sys.modules, "tracker", tracker)
    monkeypatch.setitem(sys.modules, "tracker.matching", matching)
    monkeypatch.setitem(sys.modules, "tracker.kalman_filter", kalman_filter)

    spec = importlib.util.spec_from_file_location("raft_uav_association_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_nwd_distance_is_zero_for_identical_boxes_and_increases_with_offset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper(_write_helper(tmp_path), monkeypatch)

    track = _Track((10.0, 10.0, 8.0, 8.0))
    identical = _Track((10.0, 10.0, 8.0, 8.0))
    offset = _Track((30.0, 10.0, 8.0, 8.0))
    distances = helper.nwd_distance([track], [identical, offset], 20.0)

    assert distances.shape == (1, 2)
    assert distances[0, 0] == pytest.approx(0.0)
    assert 0.0 < distances[0, 1] < 1.0


def test_motion_gate_rejects_innovation_outlier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper(_write_helper(tmp_path), monkeypatch)

    class _Kalman:
        @staticmethod
        def gating_distance(mean, covariance, measurements, only_position, metric):
            del mean, covariance, only_position, metric
            return np.asarray([1.0, 100.0])[: len(measurements)]

    tracks = [_Track((0.0, 0.0, 8.0, 8.0))]
    detections = [
        _Track((0.0, 0.0, 8.0, 8.0)),
        _Track((20.0, 0.0, 8.0, 8.0)),
    ]
    costs = helper.association_distance(
        _Kalman(),
        tracks,
        detections,
        with_reid=False,
        fuse_score=False,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        mode="gated-weighted",
        nwd_weight=0.5,
        nwd_scale=20.0,
        appearance_weight=0.25,
        appearance_min_side=16.0,
        motion_gate=True,
    )

    assert costs[0, 0] < 1.0
    assert costs[0, 1] == pytest.approx(1.0e6)


def test_legacy_mode_retains_minimum_iou_or_reid_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper(_write_helper(tmp_path), monkeypatch)

    track = _Track((0.0, 0.0, 20.0, 20.0))
    detection = _Track((4.0, 0.0, 20.0, 20.0))
    costs = helper.association_distance(
        object(),
        [track],
        [detection],
        with_reid=True,
        fuse_score=False,
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        mode="legacy-min",
        nwd_weight=0.5,
        nwd_scale=20.0,
        appearance_weight=0.25,
        appearance_min_side=16.0,
        motion_gate=True,
    )

    # Fake appearance distance is zero, so legacy min() must bypass geometry.
    assert costs[0, 0] == pytest.approx(0.0)
