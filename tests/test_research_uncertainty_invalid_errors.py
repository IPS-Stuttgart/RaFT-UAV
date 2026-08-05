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


@pytest.mark.parametrize(
    "errors",
    [
        np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        np.array([1.0 + 0.0j, 2.0 + 0.0j]),
        [True, 2.0],
        np.array([False, True]),
        np.array([np.array(True, dtype=object), 2.0], dtype=object),
    ],
)
def test_conformal_radius_rejects_lossy_calibration_errors(errors: object) -> None:
    with pytest.raises(
        ValueError,
        match="errors_m must contain real scalar values",
    ):
        fit_conformal_radius(errors, alpha=0.5)


def test_conformal_radius_ignores_masked_complex_payloads() -> None:
    errors = np.ma.array(
        np.array([1.0 + 2.0j, 3.0], dtype=object),
        mask=[True, False],
    )

    fitted = fit_conformal_radius(errors, alpha=0.5)

    assert fitted.radius_m == 3.0
    assert fitted.sample_count == 1


@pytest.mark.parametrize(
    "alpha",
    [
        True,
        np.bool_(False),
        np.complex64(0.1 + 0.8j),
        np.array([0.1]),
    ],
)
def test_conformal_radius_rejects_lossy_alpha_values(alpha: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"alpha must be a finite real scalar in \(0, 1\)",
    ):
        fit_conformal_radius([1.0, 2.0], alpha=alpha)  # type: ignore[arg-type]


def test_conformal_radius_preserves_valid_scalar_like_values() -> None:
    errors = np.array([np.array(1.0), "2.0", np.nan], dtype=object)

    fitted = fit_conformal_radius(
        errors,
        alpha=np.array(0.5),  # type: ignore[arg-type]
    )

    assert fitted.radius_m == 2.0
    assert fitted.alpha == 0.5
    assert fitted.sample_count == 2
