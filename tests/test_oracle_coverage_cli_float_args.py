from __future__ import annotations

import argparse
from collections.abc import Callable

import pytest

from raft_uav.oracle_coverage_cli import (
    _optional_positive_float,
    _optional_threshold,
    _positive_float,
)


@pytest.mark.parametrize(
    "parser",
    [_optional_threshold, _optional_positive_float, _positive_float],
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_oracle_coverage_cli_float_helpers_reject_nonfinite(
    parser: Callable[[str], float | None],
    value: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="finite"):
        parser(value)


def test_oracle_coverage_optional_float_helpers_preserve_disable_semantics() -> None:
    assert _optional_threshold("-0.1") is None
    assert _optional_positive_float("0") is None
    assert _optional_threshold("0.5") == 0.5
    assert _optional_threshold("1") == 1.0
    assert _optional_positive_float("2.5") == 2.5


def test_oracle_coverage_optional_threshold_rejects_probabilities_above_one() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="between 0 and 1"):
        _optional_threshold("1.01")


def test_oracle_coverage_positive_float_rejects_nonpositive_values() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="> 0"):
        _positive_float("0")
