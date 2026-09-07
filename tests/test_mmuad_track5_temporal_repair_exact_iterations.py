from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.track5_temporal_repair import _validate_iterations


_LARGE_EXACT_INTEGER = 2**53 + 1


def _require_extended_longdouble() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("np.longdouble has no precision beyond float64 on this platform")


def test_validate_iterations_preserves_extended_precision_integer() -> None:
    _require_extended_longdouble()
    value = np.longdouble(str(_LARGE_EXACT_INTEGER))

    assert float(value) != _LARGE_EXACT_INTEGER
    assert _validate_iterations(value) == _LARGE_EXACT_INTEGER


def test_validate_iterations_rejects_fraction_rounded_by_float() -> None:
    _require_extended_longdouble()
    value = np.longdouble("9007199254740993.5")

    assert float(value).is_integer()
    with pytest.raises(ValueError, match="iterations must be an exact positive integer"):
        _validate_iterations(value)
