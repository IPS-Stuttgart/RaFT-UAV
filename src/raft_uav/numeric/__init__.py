"""Small numeric conversion helpers shared across RaFT-UAV modules."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from numbers import Integral, Rational

import numpy as np


def optional_float(value: object) -> float | None:
    """Return a finite scalar float or ``None`` for absent/malformed values."""

    value = _scalar_numeric_input(value)
    if value is None:
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_int(value: object) -> int | None:
    """Return an integer-like finite scalar value as an int, or ``None`` otherwise."""

    value = _scalar_numeric_input(value)
    if value is None:
        return None
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Rational):
        return int(value.numerator) if value.denominator == 1 else None
    if isinstance(value, Decimal):
        return _optional_decimal_int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return _optional_decimal_int(Decimal(text))
        except InvalidOperation:
            return None
    number = optional_float(value)
    if number is None:
        return None
    if not number.is_integer():
        return None
    return int(number)


def _optional_decimal_int(value: Decimal) -> int | None:
    if not value.is_finite() or value != value.to_integral_value():
        return None
    return int(value)


def _scalar_numeric_input(value: object) -> object | None:
    if value is None:
        return None
    if _is_masked_value(value):
        return None
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, complex | np.complexfloating):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim > 0:
            return None
        value = value.item()
        if _is_masked_value(value):
            return None
        if isinstance(value, bool | np.bool_):
            return None
        if isinstance(value, complex | np.complexfloating):
            return None
    elif _is_non_scalar_array_like(value):
        return None
    return value


def _is_non_scalar_array_like(value: object) -> bool:
    """Return whether ``value`` advertises array-like, non-scalar dimensionality."""

    ndim = getattr(value, "ndim", None)
    if ndim is None:
        return False
    try:
        return int(ndim) > 0
    except (TypeError, ValueError):
        return False


def _is_masked_value(value: object) -> bool:
    """Return whether ``value`` is a masked NumPy scalar or masked array."""

    try:
        return bool(np.ma.is_masked(value))
    except (TypeError, ValueError):
        return False
