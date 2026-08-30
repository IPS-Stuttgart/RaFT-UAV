from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.research.tracklet_models import StandardizedLogisticModel


def test_predict_proba_recovers_representable_logit_after_centering_overflow() -> None:
    magnitude = np.ldexp(1.0, 1023)
    reciprocal_weight = np.ldexp(1.0, -1023)
    model = StandardizedLogisticModel(
        feature_names=("feature",),
        mean=np.array([magnitude]),
        scale=np.array([1.0]),
        weights=np.array([reciprocal_weight]),
        intercept=0.0,
    )
    frame = pd.DataFrame({"feature": [-magnitude]})

    with np.errstate(all="raise"):
        probability = model.predict_proba(frame)

    expected = 1.0 / (1.0 + np.exp(2.0))
    np.testing.assert_allclose(probability, [expected], rtol=1.0e-15, atol=0.0)


def test_predict_proba_ignores_overflowed_feature_with_zero_weight() -> None:
    magnitude = np.ldexp(1.0, 1023)
    intercept = 0.25
    model = StandardizedLogisticModel(
        feature_names=("feature",),
        mean=np.array([magnitude]),
        scale=np.array([1.0]),
        weights=np.array([0.0]),
        intercept=intercept,
    )
    frame = pd.DataFrame({"feature": [-magnitude]})

    with np.errstate(all="raise"):
        probability = model.predict_proba(frame)

    expected = 1.0 / (1.0 + np.exp(-intercept))
    np.testing.assert_allclose(probability, [expected], rtol=0.0, atol=0.0)
