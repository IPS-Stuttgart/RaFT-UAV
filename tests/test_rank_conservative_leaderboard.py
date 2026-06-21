from __future__ import annotations

import argparse

import pytest

from scripts.rank_conservative_leaderboard import _parse_constraint


def test_parse_constraint_strips_column_and_parses_numeric_value() -> None:
    assert _parse_constraint(" coverage :ge:0.95") == ("coverage", ("ge", 0.95))


def test_parse_constraint_rejects_empty_column() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="column must be non-empty"):
        _parse_constraint(":ge:0.95")


def test_parse_constraint_rejects_non_numeric_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="value must be numeric"):
        _parse_constraint("coverage:ge:not-a-number")
