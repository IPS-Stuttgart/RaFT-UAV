from __future__ import annotations

import numpy as np
import pytest

from raft_uav.multi_uav_lts._records import (
    validate_nonnegative_finite,
    validate_nonnegative_int,
    validate_unit_interval,
)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.complex64(1.0 + 0.0j),
        np.array([1.0]),
        np.ma.masked,
    ],
)
def test_nonnegative_finite_rejects_malformed_scalars(value: object) -> None:
    with pytest.raises(ValueError, match="finite non-negative scalar"):
        validate_nonnegative_finite(value, name="control")


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.complex64(1.0 + 0.0j),
        np.array([1]),
        np.ma.masked,
    ],
)
def test_nonnegative_integer_rejects_malformed_scalars(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        validate_nonnegative_int(value, name="control")


def test_multi_uav_lts_numeric_controls_accept_exact_real_scalars() -> None:
    assert validate_unit_interval(np.float64(0.5), name="control") == 0.5
    assert validate_nonnegative_finite(np.array(1.25), name="control") == 1.25
    assert validate_nonnegative_int(np.int64(2), name="control") == 2
    assert validate_nonnegative_int(np.array(3), name="control") == 3
