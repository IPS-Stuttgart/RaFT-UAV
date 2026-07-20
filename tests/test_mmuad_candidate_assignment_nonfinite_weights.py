from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_assignment_diagnostics import _assignment_weights


def test_assignment_weights_ignore_nonfinite_probability_mass() -> None:
    group = pd.DataFrame({"mixture_final_weight": [np.inf, 3.0, np.nan, -1.0]})

    weights = _assignment_weights(group)

    np.testing.assert_allclose(weights, np.array([0.0, 1.0, 0.0, 0.0]))
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)


def test_assignment_weights_fall_back_to_uniform_when_all_are_invalid() -> None:
    group = pd.DataFrame({"mixture_final_weight": [np.inf, np.nan, -2.0]})

    weights = _assignment_weights(group)

    np.testing.assert_allclose(weights, np.full(3, 1.0 / 3.0))
    assert np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1.0)
