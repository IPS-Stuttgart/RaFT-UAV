from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


def _load_helper(monkeypatch: pytest.MonkeyPatch):
    tracker = ModuleType("tracker")
    matching = ModuleType("tracker.matching")
    kalman_filter = ModuleType("tracker.kalman_filter")
    kalman_filter.chi2inv95 = {4: 9.4877}

    def iou_distance(tracks, detections):
        return np.full((len(tracks), len(detections)), 0.2, dtype=float)

    matching.iou_distance = iou_distance
    matching.fuse_score = lambda matrix, detections: np.asarray(matrix, dtype=float)
    matching.embedding_distance = lambda tracks, detections: np.zeros(
        (len(tracks), len(detections)),
        dtype=float,
    )
    tracker.matching = matching
    tracker.kalman_filter = kalman_filter
    monkeypatch.setitem(sys.modules, "tracker", tracker)
    monkeypatch.setitem(sys.modules, "tracker.matching", matching)
    monkeypatch.setitem(sys.modules, "tracker.kalman_filter", kalman_filter)

    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "raft_uav"
        / "multi_uav_lts"
        / "_upstream_association_template.py"
    )
    spec = importlib.util.spec_from_file_location("raft_uav_test_association", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Encoder:
    def __init__(self) -> None:
        self.boxes: list[np.ndarray] = []

    def inference(self, image, detections):
        boxes = np.asarray(detections, dtype=float)
        self.boxes.append(boxes.copy())
        widths = boxes[:, 2] - boxes[:, 0]
        return np.stack((widths, np.ones_like(widths)), axis=1)


def test_multiscale_reid_expands_crops_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper(monkeypatch)
    encoder = _Encoder()
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    boxes = np.asarray([[40.0, 40.0, 60.0, 60.0]])

    features = helper.multiscale_reid_features(
        encoder,
        image,
        boxes,
        (1.0, 2.0),
    )

    assert len(encoder.boxes) == 2
    assert encoder.boxes[1][0, 0] < encoder.boxes[0][0, 0]
    assert encoder.boxes[1][0, 2] > encoder.boxes[0][0, 2]
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0)


def test_anchor_feature_can_override_drifted_rolling_feature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper(monkeypatch)
    track = SimpleNamespace(
        smooth_feat=np.asarray([0.0, 1.0]),
        anchor_feat=np.asarray([1.0, 0.0]),
        to_xywh=lambda: np.asarray([10.0, 10.0, 4.0, 4.0]),
        mean=np.zeros(8),
        covariance=np.eye(8),
    )
    detection = SimpleNamespace(
        curr_feat=np.asarray([1.0, 0.0]),
        tlwh=np.asarray([8.0, 8.0, 4.0, 4.0]),
        to_xywh=lambda: np.asarray([10.0, 10.0, 4.0, 4.0]),
    )

    rolling = helper.association_distance(
        None,
        [track],
        [detection],
        with_reid=True,
        fuse_score=False,
        proximity_thresh=1.0,
        appearance_thresh=1.0,
        mode="gated-weighted",
        nwd_weight=0.0,
        nwd_scale=20.0,
        appearance_weight=1.0,
        appearance_min_side=1.0,
        motion_gate=False,
        anchor_weight=0.0,
    )
    anchored = helper.association_distance(
        None,
        [track],
        [detection],
        with_reid=True,
        fuse_score=False,
        proximity_thresh=1.0,
        appearance_thresh=1.0,
        mode="gated-weighted",
        nwd_weight=0.0,
        nwd_scale=20.0,
        appearance_weight=1.0,
        appearance_min_side=1.0,
        motion_gate=False,
        anchor_weight=1.0,
    )

    assert rolling[0, 0] == pytest.approx(0.5)
    assert anchored[0, 0] == pytest.approx(0.0)


def test_phase_schedule_interpolates_anchor_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = _load_helper(monkeypatch)
    track = SimpleNamespace(
        smooth_feat=np.asarray([0.0, 1.0]),
        anchor_feat=np.asarray([1.0, 0.0]),
        to_xywh=lambda: np.asarray([10.0, 10.0, 4.0, 4.0]),
        mean=np.zeros(8),
        covariance=np.eye(8),
    )
    detection = SimpleNamespace(
        curr_feat=np.asarray([1.0, 0.0]),
        tlwh=np.asarray([8.0, 8.0, 4.0, 4.0]),
        to_xywh=lambda: np.asarray([10.0, 10.0, 4.0, 4.0]),
    )

    early = helper.association_distance(
        None,
        [track],
        [detection],
        with_reid=True,
        fuse_score=False,
        proximity_thresh=1.0,
        appearance_thresh=1.0,
        mode="gated-weighted",
        nwd_weight=0.0,
        nwd_scale=20.0,
        appearance_weight=1.0,
        appearance_min_side=1.0,
        motion_gate=False,
        anchor_weight=0.0,
        anchor_weight_late=1.0,
        phase=0.0,
    )
    late = helper.association_distance(
        None,
        [track],
        [detection],
        with_reid=True,
        fuse_score=False,
        proximity_thresh=1.0,
        appearance_thresh=1.0,
        mode="gated-weighted",
        nwd_weight=0.0,
        nwd_scale=20.0,
        appearance_weight=1.0,
        appearance_min_side=1.0,
        motion_gate=False,
        anchor_weight=0.0,
        anchor_weight_late=1.0,
        phase=1.0,
    )

    assert early[0, 0] == pytest.approx(0.5)
    assert late[0, 0] == pytest.approx(0.0)


def test_crop_scale_parser_rejects_duplicates() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_multi_uav_lts_competition_tracker.py"
    )
    spec = importlib.util.spec_from_file_location("raft_uav_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(Exception, match="without duplicates"):
        module._crop_scales("1.0,1.0")
    assert module._crop_scales("1.0,1.25") == (1.0, 1.25)
