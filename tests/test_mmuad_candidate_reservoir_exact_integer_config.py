from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad._candidate_reservoir_config_validation_patch import (
    _exact_integer_scalar,
)


_LARGE_EXACT_INTEGER = 2**53 + 1


@pytest.mark.parametrize(
    "value",
    [
        _LARGE_EXACT_INTEGER,
        np.uint64(_LARGE_EXACT_INTEGER),
    ],
)
def test_exact_integer_scalar_preserves_values_above_float_precision(value: object) -> None:
    result = _exact_integer_scalar(value, name="count")

    assert result == _LARGE_EXACT_INTEGER


def test_exact_integer_scalar_preserves_extended_precision_integer() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("np.longdouble has no precision beyond float64 on this platform")

    value = np.longdouble(str(_LARGE_EXACT_INTEGER))

    assert float(value) != _LARGE_EXACT_INTEGER
    assert _exact_integer_scalar(value, name="count") == _LARGE_EXACT_INTEGER


def test_exact_integer_scalar_rejects_fraction_rounded_by_float() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("np.longdouble has no precision beyond float64 on this platform")

    value = np.longdouble("9007199254740993.5")

    assert float(value).is_integer()
    with pytest.raises(ValueError, match="count must be an exact integer scalar"):
        _exact_integer_scalar(value, name="count")
