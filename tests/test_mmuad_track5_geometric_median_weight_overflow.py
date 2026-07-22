from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_geometric_median_ensemble import (
    _validated_runtime_inputs,
    weighted_geometric_median,
)


def test_weighted_geometric_median_scales_large_finite_weights() -> None:
    center, iterations, displacement = weighted_geometric_median(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
        np.asarray([1.0e308, 1.0e308, 1.0e308]),
    )

    assert np.isfinite(center).all()
    assert center == pytest.approx([1.0, 0.0, 0.0])
    assert iterations >= 1
    assert displacement == pytest.approx(0.0)


def test_runtime_inputs_preserve_large_weight_ratios_without_overflow() -> None:
    estimates = pd.DataFrame()

    normalized = _validated_runtime_inputs(
        [
            ("primary", estimates, 1.0e308),
            ("secondary", estimates, 5.0e307),
        ]
    )

    assert [weight for _, _, weight in normalized] == pytest.approx([1.0, 0.5])
