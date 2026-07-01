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
    if _is_non_scalar_array_like(value):
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


def _is_non_scalar_array_like(value: object) -> bool:
    """Return whether ``value`` advertises array-like, non-scalar dimensionality."""

    ndim = getattr(value, "ndim", None)
    if ndim is None:
        return False
    try:
        return int(ndim) > 0
    except (TypeError, ValueError):
        return False
