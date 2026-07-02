"""Small numeric conversion helpers shared across RaFT-UAV modules."""

from __future__ import annotations

import math

import numpy as np


def optional_float(value: object) -> float | None:
    """Return a finite scalar float or ``None`` for absent/malformed values."""

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
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_int(value: object) -> int | None:
    """Return an integer-like finite scalar value as an int, or ``None`` otherwise."""

    exact = _optional_exact_int(value)
    if exact is not None:
        return exact
    number = optional_float(value)
    if number is None:
        return None
    if not number.is_integer():
        return None
    return int(number)


def _optional_exact_int(value: object) -> int | None:
    """Return exact integer scalars before float conversion can round them."""

    if value is None:
        return None
    if _is_masked_value(value):
        return None
    if isinstance(value, bool | np.bool_):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim > 0:
            return None
        value = value.item()
        if _is_masked_value(value) or isinstance(value, bool | np.bool_):
            return None
    elif _is_non_scalar_array_like(value):
        return None
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 10)
        except ValueError:
            return None
    return None


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
