"""Small numeric conversion helpers shared across RaFT-UAV modules."""

from __future__ import annotations

import math

import numpy as np


def optional_float(value: object) -> float | None:
    """Return a finite float or ``None`` for absent/malformed values."""

    if value is None:
        return None
    if isinstance(value, bool | np.bool_):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_int(value: object) -> int | None:
    """Return an integer converted from a finite numeric value, if available."""

    number = optional_float(value)
    return None if number is None else int(number)
