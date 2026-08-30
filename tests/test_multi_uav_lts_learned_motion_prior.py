import json

import numpy as np

from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.learned_motion_prior import (
    fit_prior,
    load_sequence_translations,
    map_affinity,
    make_affinity,
    transition_features,
)
from raft_uav.multi_uav_lts.scene_stabilization import make_stabilized_geometry


def detection(frame: int, center_x: float):
    return Detection(frame, 1, center_x - 3, 17, 6, 6, 0.9, 1, 1.0)


def test_motion_prior_prefers_training_like_transition():
    examples = []
    for index in range(200):
        examples.append(
            transition_features(
                detection(1, 20 + 0.01 * index),
                detection(2, 23 + 0.01 * index),
            )
        )
    prior = fit_prior(np.asarray(examples), components=1)
    assert prior.similarity(detection(1, 20), detection(2, 23)) > prior.similarity(
        detection(1, 20),
        detection(2, 80),
    )


def test_motion_affinity_can_be_evaluated_in_stabilized_coordinates():
    features = np.asarray(
        [transition_features(detection(1, 20 + index / 100), detection(2, 20 + index / 100))
         for index in range(100)]
    )
    prior = fit_prior(features, components=1)
    affinity = make_affinity(prior)
    geometry = make_stabilized_geometry({1: (0, 0), 2: (0, -20)})
    mapped = map_affinity(affinity, geometry)
    assert mapped(detection(1, 20), detection(2, 40)) > affinity(
        detection(1, 20),
        detection(2, 40),
    )


def test_per_sequence_stabilization_cache_loader(tmp_path):
    payload = {
        "format": "raft-uav-sequence-stabilization-v1",
        "sequence": "seq-a",
        "translations": {"1": [0, 0], "2": [1.5, -2.0]},
    }
    (tmp_path / "seq-a.json").write_text(json.dumps(payload), encoding="utf-8")
    translations = load_sequence_translations(tmp_path, "seq-a")
    assert translations[2] == (1.5, -2.0)
