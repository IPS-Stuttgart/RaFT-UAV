from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from raft_uav.multi_uav_lts import _proposal_graph_core as core
from raft_uav.multi_uav_lts import _proposal_similarity_motion as similarity
from raft_uav.multi_uav_lts import experimental_proposal_graph_tracker as experimental
from raft_uav.multi_uav_lts import proposal_graph_tracker
from raft_uav.multi_uav_lts._records import Detection


def _row(
    frame: int,
    object_id: int,
    center_x: float,
    center_y: float,
    *,
    width: float = 10.0,
    height: float = 8.0,
) -> Detection:
    return Detection(
        frame,
        object_id,
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        width,
        height,
        0.9,
        1,
        1.0,
    )


def _parameters(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "enable_common_motion": True,
        "common_motion_min_pairs": 4,
        "common_motion_max_normalized_step": 50.0,
        "common_motion_max_normalized_residual": 3.0,
        "center_weight": 1.0,
        "size_weight": 0.5,
        "iou_weight": 0.5,
        "velocity_weight": 0.5,
        "gap_weight": 0.1,
        "confidence_weight": 0.1,
        "max_link_gap": 30,
        "max_link_cost": 5.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _transformed_nodes(
    *,
    scale: float,
    angle_deg: float,
    tx: float,
    ty: float,
) -> tuple[core._Node, ...]:
    points = (
        (100.0, 100.0),
        (210.0, 90.0),
        (320.0, 145.0),
        (145.0, 260.0),
        (290.0, 285.0),
        (410.0, 225.0),
        (510.0, 155.0),
        (445.0, 335.0),
    )
    angle = math.radians(angle_deg)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    nodes: list[core._Node] = []
    for object_id, (x_value, y_value) in enumerate(points, start=1):
        left = _row(1, object_id, x_value, y_value)
        right_x = scale * (cosine * x_value - sine * y_value) + tx
        right_y = scale * (sine * x_value + cosine * y_value) + ty
        right = _row(
            2,
            object_id,
            right_x,
            right_y,
            width=10.0 * scale,
            height=8.0 * scale,
        )
        nodes.append(core._Node(len(nodes), left))
        nodes.append(core._Node(len(nodes), right))
    return tuple(nodes)


def test_similarity_motion_recovers_reliable_scale_rotation_translation() -> None:
    config = similarity.SimilarityMotionConfig(
        min_pairs=4,
        max_scale_change=0.2,
        max_rotation_deg=10.0,
        max_normalized_residual=0.5,
        min_normalized_spread=2.0,
        min_residual_improvement=0.02,
    )
    estimates = similarity.estimate_common_motion(
        _transformed_nodes(scale=1.05, angle_deg=4.0, tx=12.0, ty=-7.0),
        _parameters(),
        config=config,
    )

    transform = estimates[1]
    assert transform.model == "similarity"
    assert transform.support == 8
    assert transform.scale == pytest.approx(1.05, abs=1e-8)
    assert math.degrees(transform.angle_rad) == pytest.approx(4.0, abs=1e-8)
    assert transform.tx == pytest.approx(12.0, abs=1e-8)
    assert transform.ty == pytest.approx(-7.0, abs=1e-8)


def test_degenerate_similarity_fit_falls_back_to_exact_translation() -> None:
    points = ((100.0, 100.0), (105.0, 100.0), (100.0, 105.0), (105.0, 105.0))
    nodes: list[core._Node] = []
    for object_id, (x_value, y_value) in enumerate(points, start=1):
        nodes.append(core._Node(len(nodes), _row(1, object_id, x_value, y_value)))
        nodes.append(
            core._Node(
                len(nodes),
                _row(2, object_id, x_value + 7.0, y_value - 3.0),
            )
        )
    config = similarity.SimilarityMotionConfig(
        min_pairs=4,
        min_normalized_spread=10.0,
    )
    transform = similarity.estimate_common_motion(
        tuple(nodes),
        _parameters(),
        config=config,
    )[1]

    assert transform.model == "translation"
    assert tuple(transform) == pytest.approx((7.0, -3.0))
    assert transform.scale == 1.0
    assert transform.cos_theta == 1.0
    assert transform.sin_theta == 0.0


def test_similarity_prediction_composes_motion_and_residual_velocity() -> None:
    step = similarity.SimilarityTransform(
        scale=1.02,
        cos_theta=math.cos(math.radians(3.0)),
        sin_theta=math.sin(math.radians(3.0)),
        tx=5.0,
        ty=-2.0,
        model="similarity",
    )
    first = _row(1, 1, 100.0, 80.0)
    common_second = step.apply_detection(first, 2)
    second = Detection(
        common_second.frame_id,
        common_second.object_id,
        common_second.x1 + 2.0,
        common_second.y1 - 1.0,
        common_second.width,
        common_second.height,
        common_second.confidence,
        common_second.class_id,
        common_second.visibility,
    )
    prediction = similarity.predict((first, second), 3, {1: step, 2: step})
    common_third = step.apply_detection(second, 3)
    residual_x, residual_y = step.apply_vector(2.0, -1.0)

    assert prediction.center_x == pytest.approx(common_third.center_x + residual_x)
    assert prediction.center_y == pytest.approx(common_third.center_y + residual_y)
    assert np.isfinite(
        [prediction.x1, prediction.y1, prediction.width, prediction.height]
    ).all()


def test_experimental_cli_installs_and_restores_similarity_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_estimator = core._estimate_common_motion
    original_predict = core._predict
    seen: dict[str, object] = {}

    def fake_main(arguments: list[str]) -> int:
        seen["arguments"] = tuple(arguments)
        seen["estimator"] = core._estimate_common_motion
        seen["predict"] = core._predict
        return 17

    monkeypatch.setattr(proposal_graph_tracker, "main", fake_main)
    result = experimental.main(
        [
            "--no-sequence-cache",
            "--common-motion-model",
            "similarity",
            "--enable-common-motion",
        ]
    )

    assert result == 17
    assert seen["arguments"] == ("--enable-common-motion",)
    assert seen["estimator"] is not original_estimator
    assert seen["predict"] is similarity.predict
    assert core._estimate_common_motion is original_estimator
    assert core._predict is original_predict
