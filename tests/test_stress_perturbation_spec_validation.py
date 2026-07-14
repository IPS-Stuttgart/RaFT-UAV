from __future__ import annotations

import numpy as np
import pytest

from raft_uav.experiments.stress_perturbation_suite import PerturbationSpec


@pytest.mark.parametrize(
    "field",
    [
        "time_jitter_std_s",
        "velocity_noise_std_mps",
        "position_noise_std_m",
        "catprob_scale",
    ],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_perturbation_spec_rejects_nonfinite_scales(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=rf"{field} must be finite and nonnegative"):
        PerturbationSpec(name="invalid", **{field: value})


def test_perturbation_spec_accepts_finite_nonnegative_scales() -> None:
    spec = PerturbationSpec(
        name="valid",
        time_jitter_std_s=0.1,
        velocity_noise_std_mps=0.2,
        position_noise_std_m=0.3,
        catprob_scale=1.5,
    )

    assert spec.time_jitter_std_s == pytest.approx(0.1)
    assert spec.velocity_noise_std_mps == pytest.approx(0.2)
    assert spec.position_noise_std_m == pytest.approx(0.3)
    assert spec.catprob_scale == pytest.approx(1.5)
