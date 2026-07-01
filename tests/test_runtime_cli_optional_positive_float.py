import pytest

from raft_uav.runtime_cli_config import _optional_positive_float


def test_optional_positive_float_rejects_negative_values():
    with pytest.raises(ValueError, match="range_m must be nonnegative"):
        _optional_positive_float(-1.0, "range_m")


def test_optional_positive_float_treats_zero_as_disabled():
    assert _optional_positive_float(0.0, "range_m") is None
