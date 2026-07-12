"""Shared argparse value parsers for MMUAD command-line interfaces."""

from __future__ import annotations

import argparse
import math


def nonnegative_finite_float(value: str) -> float:
    """Parse a finite floating-point value greater than or equal to zero."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected a finite nonnegative float") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative float")
    return parsed
