from __future__ import annotations

import math

import pytest

from raft_uav.multi_uav_lts._proposal_edge_likelihood import (
    EDGE_FEATURE_NAMES,
    EdgeLikelihoodModel,
)
from raft_uav.multi_uav_lts._proposal_metric_edge_likelihood import (
    MetricEdgeLikelihoodModel,
)


def _head(intercept: float) -> EdgeLikelihoodModel:
    return EdgeLikelihoodModel(
        schema="raft-uav-multi-uav-lts-edge-likelihood-v1",
        feature_names=EDGE_FEATURE_NAMES,
        means=(0.0,) * len(EDGE_FEATURE_NAMES),
        scales=(1.0,) * len(EDGE_FEATURE_NAMES),
        coefficients=(0.0,) * len(EDGE_FEATURE_NAMES),
        intercept=intercept,
        training_example_count=20,
        positive_example_count=10,
        negative_example_count=10,
        sequence_count=4,
        l2_penalty=1.0,
        metadata={},
    )


def test_metric_edge_cost_is_weighted_head_nll() -> None:
    model = MetricEdgeLikelihoodModel(
        schema="raft-uav-multi-uav-lts-metric-edge-likelihood-v1",
        identity=_head(0.0),
        hota_005=_head(math.log(3.0)),
        clear_050=_head(-math.log(3.0)),
        identity_weight=1.0,
        hota_weight=2.0,
        clear_weight=1.0,
        metadata={"selected_sequences": ["A", "B", "C", "D"]},
    )
    features = (0.0,) * len(EDGE_FEATURE_NAMES)
    expected = (
        model.identity.negative_log_probability(features)
        + 2.0 * model.hota_005.negative_log_probability(features)
        + model.clear_050.negative_log_probability(features)
    ) / 4.0
    assert model.negative_log_probability(features) == pytest.approx(expected)
    probabilities = model.head_probabilities(features)
    assert probabilities["identity"] == pytest.approx(0.5)
    assert probabilities["hota_005"] == pytest.approx(0.75)
    assert probabilities["clear_050"] == pytest.approx(0.25)


def test_metric_edge_round_trip_and_weight_override() -> None:
    model = MetricEdgeLikelihoodModel(
        schema="raft-uav-multi-uav-lts-metric-edge-likelihood-v1",
        identity=_head(0.0),
        hota_005=_head(0.5),
        clear_050=_head(-0.5),
    )
    restored = MetricEdgeLikelihoodModel.from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()
    hota_only = restored.with_weights(
        identity_weight=0.0,
        hota_weight=1.0,
        clear_weight=0.0,
    )
    features = (0.0,) * len(EDGE_FEATURE_NAMES)
    assert hota_only.negative_log_probability(features) == pytest.approx(
        hota_only.hota_005.negative_log_probability(features)
    )


def test_metric_edge_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError, match="at least one metric-edge head weight"):
        MetricEdgeLikelihoodModel(
            schema="raft-uav-multi-uav-lts-metric-edge-likelihood-v1",
            identity=_head(0.0),
            hota_005=_head(0.0),
            clear_050=_head(0.0),
            identity_weight=0.0,
            hota_weight=0.0,
            clear_weight=0.0,
        )
