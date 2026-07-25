from __future__ import annotations

import numpy as np
import pytest

from raft_uav.research.uncertainty import fit_conformal_radius


def test_conformal_radius_rejects_negative_calibration_errors() -> None:
    with pytest.raises(
        ValueError,
        match="errors_m must contain only non-negative values",
    ):
        fit_conformal_radius([-5.0, 2.0], alpha=0.9)


def test_conformal_radius_ignores_masked_negative_payloads() -> None:
    errors = np.ma.array([-5.0, 2.0], mask=[True, False])

    fitted = fit_conformal_radius(errors, alpha=0.1)

    assert fitted.radius_m == 2.0
    assert fitted.sample_count == 1
