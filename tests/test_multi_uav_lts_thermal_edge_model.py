import numpy as np
from raft_uav.multi_uav_lts._records import Detection
from raft_uav.multi_uav_lts.thermal_edge_model import fit_logistic, pair_features

def box(frame, x):
    return Detection(frame, 1, x, 10, 8, 8, 0.9, 1, 1.0)

def test_pair_features_and_logistic_separate_examples():
    image = np.zeros((40, 40), dtype=float)
    image[12:18, 12:18] = 1
    shifted = np.zeros_like(image)
    shifted[12:18, 14:20] = 1
    negative = np.random.default_rng(1).random(image.shape)
    positive = pair_features(image, box(1, 11), shifted, box(2, 13))
    bad = pair_features(image, box(1, 11), negative, box(2, 25))
    x = np.vstack([positive, positive + 0.01, bad, bad + 0.01])
    y = np.array([1, 1, 0, 0])
    model = fit_logistic(x, y, l2=0.1)
    assert model.probability(positive) > model.probability(bad)
