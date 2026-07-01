"""Small numeric conversion helpers shared across RaFT-UAV modules."""

from __future__ import annotations

import math

import numpy as np


def optional_float(value: object) -> float | None:
    """Return a finite scalar float or ``None`` for absent/malformed values."""

    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_int(value: object) -> int | None:
    """Return an integer-like finite scalar value as an int, or ``None`` otherwise."""

    number = optional_float(value)
    if number is None:
        return None
    if not number.is_integer():
        return None
    return int(number)
