from __future__ import annotations

import numpy as np
import pytest

from raft_uav.baselines.kalman import white_acceleration_process_noise


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("dt_s", True),
        ("dt_s", np.bool_(False)),
        ("dt_s", -0.1),
        ("dt_s", np.nan),
        ("dt_s", np.inf),
        ("dt_s", [0.5]),
        ("acceleration_std", True),
        ("acceleration_std", np.bool_(False)),
        ("acceleration_std", -0.1),
        ("acceleration_std", np.nan),
        ("acceleration_std", np.inf),
        ("acceleration_std", np.asarray([4.0])),
    ],
)
def test_process_noise_rejects_invalid_controls(
    parameter: str,
    value: object,
) -> None:
    arguments = {"dt_s": 0.5, "acceleration_std": 4.0}
    arguments[parameter] = value

    with pytest.raises(
        ValueError,
        match=rf"{parameter} must be a finite non-negative real scalar",
    ):
        white_acceleration_process_noise(**arguments)


def test_process_noise_accepts_zero_dimensional_nonnegative_controls() -> None:
    covariance = white_acceleration_process_noise(
        np.asarray(0.5),
        np.asarray(4.0),
    )
    zero_covariance = white_acceleration_process_noise(0.0, 0.0)

    np.testing.assert_allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    np.testing.assert_array_equal(zero_covariance, np.zeros((6, 6)))
