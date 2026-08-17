"""Lossless numeric parsing helpers for runtime configuration values."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation
from numbers import Integral, Real


def optional_float(value: object) -> float | None:
    """Return a finite float for a scalar numeric value, or ``None`` if invalid.

    Boolean values are rejected explicitly because ``bool`` is an ``int`` subclass.
    Numeric strings are accepted to preserve values coming from environment/config
    serialization, while non-finite values are rejected.
    """

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, Real):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except (TypeError, ValueError, OverflowError):
            return None
    else:
        return None

    return number if math.isfinite(number) else None


def optional_int(value: object) -> int | None:
    """Return an exact integer representation, or ``None`` if conversion is lossy.

    Integral floats and decimal strings such as ``8.0`` and ``"24.0"`` are
    accepted. Fractional values and booleans are rejected rather than being
    silently truncated by :class:`int`.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            return None
        return int(number)
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    return int(number)
