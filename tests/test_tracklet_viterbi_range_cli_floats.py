from __future__ import annotations

import argparse

import pytest

from raft_uav.tracklet_viterbi_range_cli import _nonnegative_float


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_range_cli_rejects_nonfinite_nonnegative_float(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        _nonnegative_float(value)


@pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("2.5", 2.5)])
def test_range_cli_accepts_finite_nonnegative_float(value: str, expected: float) -> None:
    assert _nonnegative_float(value) == expected
