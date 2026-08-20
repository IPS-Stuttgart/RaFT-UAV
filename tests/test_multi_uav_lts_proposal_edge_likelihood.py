from __future__ import annotations

from pathlib import Path

import pytest

from raft_uav.multi_uav_lts._proposal_edge_likelihood import (
    EDGE_FEATURE_NAMES,
    build_edge_feature_context,
    fit_edge_likelihood,
    load_edge_likelihood_model,
    write_edge_likelihood_model,
)
from raft_uav.multi_uav_lts._records import Detection, format_detection
from raft_uav.multi_uav_lts.experimental_proposal_graph_tracker import (
    main as experimental_main,
)
from raft_uav.multi_uav_lts.proposal_edge_model import fit_edge_model_from_lts


def row(
    frame: int,
    object_id: int,
    center_x: float,
    center_y: float,
    *,
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        frame,
        object_id,
        center_x - 2.0,
        center_y - 2.0,
        4.0,
        4.0,
        confidence,
        1,
        1.0,
    )


def write_rows(path: Path, rows: list[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(format_detection(item) + "\n" for item in rows),
        encoding="utf-8",
    )


def test_swarm_descriptor_prefers_constellation_preserving_edge() -> None:
    rows = (
        row(1, 1, 0.0, 0.0),
        row(1, 2, 10.0, 0.0),
        row(1, 3, 0.0, 10.0),
        row(2, 11, 5.0, 0.0),
        row(2, 12, 15.0, 0.0),
        row(2, 13, 5.0, 10.0),
    )
    context = build_edge_feature_context(
        rows,
        neighbor_count=2,
        radius_scale=10.0,
    )

    correct = context.swarm_features(rows[0], rows[3])
    wrong = context.swarm_features(rows[0], rows[4])

    assert correct[0] < wrong[0]
    assert correct[1] > 0.0


def test_edge_likelihood_fit_and_json_roundtrip(tmp_path: Path) -> None:
    positive = (0.1, 0.05, 0.1, 0.05, 0.0, 0.1, 1.0, 0.0)
    negative = (1.5, 0.8, 0.9, 0.1, 0.0, 1.5, 1.0, 0.2)
    features = [positive, negative] * 12
    labels = [1, 0] * 12
    sequences = ["A", "B"] * 12

    model = fit_edge_likelihood(
        features,
        labels,
        sequence_ids=sequences,
        l2_penalty=0.1,
    )
    assert model.feature_names == EDGE_FEATURE_NAMES
    assert model.probability(positive) > model.probability(negative)

    model_path = tmp_path / "edge-model.json"
    write_edge_likelihood_model(model, model_path)
    loaded = load_edge_likelihood_model(model_path)
    assert loaded.to_dict() == model.to_dict()


def test_fit_edge_model_from_lts_builds_hard_negative_examples(
    tmp_path: Path,
) -> None:
    truth_dir = tmp_path / "truth"
    proposal_dir = tmp_path / "proposals"
    truth_rows: list[Detection] = []
    proposal_rows: list[Detection] = []
    for frame in (1, 2, 3):
        for identity, start in enumerate((0.0, 6.0, 12.0), start=1):
            center = start + frame - 1
            truth_rows.append(row(frame, identity, center, 10.0))
            proposal_rows.append(
                row(frame, frame * 100 + identity, center, 10.0)
            )
    write_rows(truth_dir / "SEQ.txt", truth_rows)
    write_rows(proposal_dir / "SEQ.txt", proposal_rows)

    result = fit_edge_model_from_lts(
        proposal_dir,
        truth_dir,
        min_truth_iou=0.5,
        negative_candidates_per_left=2,
        l2_penalty=0.1,
    )

    assert result.positive_candidate_edges > 0
    assert result.negative_candidate_edges > 0
    positive = next(example for example in result.examples if example.label == 1)
    negative = next(example for example in result.examples if example.label == 0)
    assert result.model.probability(positive.features) > result.model.probability(
        negative.features
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--enable-delayed-path-cover", "--swarm-neighbors", "0"], "swarm_neighbors"),
        (
            ["--enable-delayed-path-cover", "--swarm-radius-scale", "nan"],
            "swarm_radius_scale",
        ),
        (["--swarm-relative-weight", "1"], "require delayed path cover"),
    ],
)
def test_experimental_edge_controls_fail_before_dataset_io(
    arguments: list[str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        experimental_main(["unused", *arguments])


def test_learned_cost_is_applied_before_successor_pruning() -> None:
    from raft_uav.multi_uav_lts import _proposal_graph_core as core
    from raft_uav.multi_uav_lts._proposal_delayed_path_cover import (
        DelayedPathCoverConfig,
        _delayed_candidates,
    )
    from raft_uav.multi_uav_lts._proposal_edge_likelihood import (
        EdgeLikelihoodModel,
        build_edge_feature_context,
    )
    from raft_uav.multi_uav_lts.proposal_graph_tracker import _parameters

    left = row(1, 1, 0.0, 0.0)
    raw_favorite = row(2, 2, 1.0, 0.0, confidence=0.1)
    learned_favorite = row(2, 3, 2.0, 0.0, confidence=1.0)
    tracklets = (
        core._Tracklet(0, (left,), (0,)),
        core._Tracklet(1, (raw_favorite,), (1,)),
        core._Tracklet(2, (learned_favorite,), (2,)),
    )
    parameters = _parameters(
        min_proposal_confidence=0.0,
        duplicate_iou=0.95,
        min_seed_iou=0.05,
        anchor_max_cost=1.25,
        anchor_min_margin=0.15,
        enable_global_links=True,
        max_link_gap=0,
        max_link_cost=5.0,
        center_weight=1.0,
        size_weight=0.25,
        iou_weight=0.35,
        velocity_weight=0.5,
        gap_weight=0.04,
        confidence_weight=0.05,
        enable_common_motion=False,
        common_motion_min_pairs=4,
        common_motion_max_normalized_step=8.0,
        common_motion_max_normalized_residual=1.5,
        interpolate_max_gap=0,
        birth_min_hits=3,
        birth_min_span=2,
        birth_min_mean_confidence=0.0,
        birth_require_border_entry=False,
        birth_min_inward_motion=0.0,
        image_width=None,
        image_height=None,
        border_margin_fraction=0.08,
        border_gap_discount=0.35,
    )
    coefficients = [0.0] * len(EDGE_FEATURE_NAMES)
    coefficients[EDGE_FEATURE_NAMES.index("confidence_deficit")] = -10.0
    model = EdgeLikelihoodModel(
        schema="raft-uav-multi-uav-lts-edge-likelihood-v1",
        feature_names=EDGE_FEATURE_NAMES,
        means=(0.0,) * len(EDGE_FEATURE_NAMES),
        scales=(1.0,) * len(EDGE_FEATURE_NAMES),
        coefficients=tuple(coefficients),
        intercept=0.0,
        training_example_count=2,
        positive_example_count=1,
        negative_example_count=1,
        sequence_count=1,
        l2_penalty=1.0,
    )
    config = DelayedPathCoverConfig(
        lookahead_frames=1,
        successors_per_frame=1,
        continuation_weight=0.0,
        edge_model_weight=1.0,
        edge_model_clip=4.0,
    )
    context = build_edge_feature_context(
        (left, raw_favorite, learned_favorite),
        neighbor_count=2,
        radius_scale=12.0,
    )

    candidates = _delayed_candidates(
        tracklets,
        parameters,
        {},
        config,
        edge_model=model,
        feature_context=context,
    )

    assert (0, 2) in candidates
    assert (0, 1) not in candidates
