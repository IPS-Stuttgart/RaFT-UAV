from __future__ import annotations

import json
import math
from dataclasses import replace

import pytest

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts._proposal_sim2 import (
    SimilarityMotionConfig,
    SimilarityMotionStep,
    SimilarityTransform,
    step_summary,
    cumulative_transforms,
    estimate_similarity_steps,
    restore_rows,
    stabilize_rows,
)
from raft_uav.multi_uav_lts.sim2_proposal_graph_tracker import (
    _delegated_arguments,
    _reject_incompatible_options,
)


def _detection(
    frame_id: int,
    object_id: int,
    center_x: float,
    center_y: float,
    *,
    width: float = 10.0,
    height: float = 8.0,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame_id,
        object_id,
        center_x - 0.5 * width,
        center_y - 0.5 * height,
        width,
        height,
        confidence,
        1,
        1.0,
    )


def _config(**updates: object) -> SimilarityMotionConfig:
    values: dict[str, object] = {
        "min_pairs": 4,
        "max_rows": 96,
        "max_normalized_step": 50.0,
        "max_normalized_residual": 0.5,
        "max_scale_deviation": 0.2,
        "max_rotation_degrees": 20.0,
        "min_spread_normalized": 2.0,
        "min_residual_improvement": 0.01,
    }
    values.update(updates)
    return SimilarityMotionConfig(**values)  # type: ignore[arg-type]


def _transform_rows(
    rows: list[Detection],
    transform: SimilarityTransform,
    frame_id: int,
) -> list[Detection]:
    return [
        transform.apply_detection(replace(row, frame_id=frame_id)) for row in rows
    ]


def _irregular_frame() -> list[Detection]:
    centers = ((20, 30), (120, 25), (35, 140), (180, 160), (245, 70), (90, 220))
    return [
        _detection(1, index + 1, center_x, center_y)
        for index, (center_x, center_y) in enumerate(centers)
    ]


def test_estimator_recovers_reliable_similarity_transform() -> None:
    left = _irregular_frame()
    angle = math.radians(6.0)
    expected = SimilarityTransform(
        1.06,
        math.cos(angle),
        math.sin(angle),
        18.0,
        -11.0,
    )
    right = _transform_rows(left, expected, 2)

    step = estimate_similarity_steps(left + right, _config())[1]

    assert step.model == "sim2"
    assert step.transform.scale == pytest.approx(1.06, abs=1e-6)
    assert step.transform.rotation_degrees == pytest.approx(6.0, abs=1e-6)
    assert step.transform.translation_x == pytest.approx(18.0, abs=1e-5)
    assert step.transform.translation_y == pytest.approx(-11.0, abs=1e-5)


def test_estimator_uses_translation_when_similarity_has_no_gain() -> None:
    left = _irregular_frame()
    expected = SimilarityTransform(1.0, 1.0, 0.0, 13.0, -4.0)
    right = _transform_rows(left, expected, 2)

    step = estimate_similarity_steps(left + right, _config())[1]

    assert step.model == "translation"
    assert step.fallback_reason == "insufficient_improvement"
    assert step.transform.translation_x == pytest.approx(13.0)
    assert step.transform.translation_y == pytest.approx(-4.0)


def test_estimator_falls_back_when_geometry_has_insufficient_spread() -> None:
    left = [
        _detection(1, index + 1, 100 + index, 100 + 0.2 * index)
        for index in range(6)
    ]
    right = _transform_rows(
        left,
        SimilarityTransform(1.0, 1.0, 0.0, 5.0, 3.0),
        2,
    )

    step = estimate_similarity_steps(
        left + right,
        _config(min_spread_normalized=10.0),
    )[1]

    assert step.model == "translation"
    assert step.fallback_reason == "insufficient_spread"


def test_stabilization_and_restoration_round_trip() -> None:
    rows = [_detection(1, 1, 20, 30), _detection(2, 1, 45, 50)]
    angle = math.radians(4.0)
    step = SimilarityMotionStep(
        1,
        SimilarityTransform(
            1.03,
            math.cos(angle),
            math.sin(angle),
            10.0,
            -5.0,
        ),
        "sim2",
        4,
        4,
        0.1,
        0.2,
        5.0,
        None,
    )
    transforms = cumulative_transforms({1: step}, 2)

    restored = restore_rows(stabilize_rows(rows, transforms), transforms)

    for expected, actual in zip(rows, restored, strict=True):
        assert actual.center_x == pytest.approx(expected.center_x, abs=1e-9)
        assert actual.center_y == pytest.approx(expected.center_y, abs=1e-9)
        assert actual.width == pytest.approx(expected.width, abs=1e-9)
        assert actual.height == pytest.approx(expected.height, abs=1e-9)


def test_similarity_composition_and_inverse_match_direct_application() -> None:
    first = SimilarityTransform(1.04, math.cos(0.1), math.sin(0.1), 3.0, -2.0)
    second = SimilarityTransform(
        0.98,
        math.cos(-0.04),
        math.sin(-0.04),
        -1.0,
        5.0,
    )
    composed = second.compose(first)
    point = (17.0, 29.0)
    intermediate = first.apply_xy(*point)

    assert composed.apply_xy(*point) == pytest.approx(
        second.apply_xy(*intermediate)
    )
    assert composed.inverse().apply_xy(*composed.apply_xy(*point)) == pytest.approx(
        point
    )


def test_sequence_summary_is_strict_json_when_every_step_is_identity() -> None:
    identity = SimilarityMotionStep(
        1,
        SimilarityTransform.identity(),
        "identity",
        0,
        0,
        math.nan,
        math.nan,
        0.0,
        "insufficient_rows",
    )

    summary = step_summary({1: identity})

    assert summary["median_residual_normalized"] is None
    json.dumps(summary, allow_nan=False)


def test_incompatible_double_motion_and_border_controls_are_rejected() -> None:
    for option in (
        "--enable-common-motion",
        "--common-motion-min-pairs=5",
        "--birth-require-border-entry",
        "--image-width=1920",
    ):
        with pytest.raises(ValueError, match="incompatible"):
            _reject_incompatible_options([option])


def test_delegated_arguments_replace_only_generated_paths(tmp_path) -> None:
    arguments = [
        "proposals",
        "--first-frame-label-dir",
        "labels",
        "--output-dir",
        "predictions",
        "--output-json=summary.json",
        "--enable-delayed-path-cover",
    ]

    delegated = _delegated_arguments(
        arguments,
        stabilized_dir=tmp_path / "stabilized",
        output_dir=tmp_path / "output",
        summary_path=tmp_path / "summary.json",
    )

    assert delegated[0] == str(tmp_path / "stabilized")
    assert delegated[delegated.index("--output-dir") + 1] == str(tmp_path / "output")
    assert "--output-json=summary.json" not in delegated
    assert delegated[delegated.index("--output-json") + 1] == str(
        tmp_path / "summary.json"
    )
    assert "--enable-delayed-path-cover" in delegated
    assert "--no-sequence-cache" in delegated
