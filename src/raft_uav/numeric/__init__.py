"""Small numeric conversion helpers shared across RaFT-UAV modules."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from numbers import Integral, Rational
from operator import index

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
    if isinstance(value, float | np.floating):
        if not np.isfinite(value) or not value.is_integer():
            return None
        return int(value)
    number = optional_float(value)
    if number is None:
        return None
    if not number.is_integer():
        return None
    return int(number)


def stable_euclidean_rate(
    current: np.ndarray,
    previous: np.ndarray,
    dt_s: float,
) -> float:
    """Return a representable finite Euclidean displacement rate when one exists."""

    current_array = np.asarray(current, dtype=float)
    previous_array = np.asarray(previous, dtype=float)
    dt_s = float(dt_s)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = current_array - previous_array
        distance = float(np.linalg.norm(delta))
        direct = float(distance / dt_s)
    if np.isfinite(direct):
        return direct
    if not (
        current_array.shape == previous_array.shape
        and bool(np.isfinite(current_array).all())
        and bool(np.isfinite(previous_array).all())
        and np.isfinite(dt_s)
        and dt_s > 0.0
    ):
        return np.nan

    scale = float(max(np.max(np.abs(current_array)), np.max(np.abs(previous_array))))
    if scale <= 0.0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_delta = current_array / scale - previous_array / scale
        scaled_distance = float(np.hypot.reduce(np.abs(scaled_delta).reshape(-1)))
    repaired = _positive_product_ratio(scale, scaled_distance, dt_s)
    return repaired if np.isfinite(repaired) else np.nan


def _positive_product_ratio(
    numerator_a: float,
    numerator_b: float,
    denominator: float,
) -> float:
    """Return ``a * b / denominator`` without overflowing intermediate products."""

    if numerator_a == 0.0 or numerator_b == 0.0:
        return 0.0
    numerator_a_mantissa, numerator_a_exponent = math.frexp(float(numerator_a))
    numerator_b_mantissa, numerator_b_exponent = math.frexp(float(numerator_b))
    denominator_mantissa, denominator_exponent = math.frexp(float(denominator))
    mantissa = (numerator_a_mantissa * numerator_b_mantissa) / denominator_mantissa
    exponent = numerator_a_exponent + numerator_b_exponent - denominator_exponent
    try:
        return math.ldexp(mantissa, exponent)
    except OverflowError:
        return math.inf


def _optional_decimal_int(value: Decimal) -> int | None:
    if not value.is_finite() or value != value.to_integral_value():
        return None
    return int(value)


def _scalar_numeric_input(value: object) -> object | None:
    seen_array_ids: set[int] = set()
    while True:
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
            array_id = id(value)
            if array_id in seen_array_ids:
                return None
            seen_array_ids.add(array_id)
            value = value.item()
            continue
        if _is_non_scalar_array_like(value):
            return None
        return value


def _is_non_scalar_array_like(value: object) -> bool:
    """Return whether ``value`` advertises array-like, non-scalar dimensionality."""

    ndim = getattr(value, "ndim", None)
    if ndim is None:
        return False
    try:
        return index(ndim) != 0
    except (OverflowError, TypeError, ValueError):
        return True


def _is_masked_value(value: object) -> bool:
    """Return whether ``value`` is a masked NumPy scalar or masked array."""

    try:
        return bool(np.ma.is_masked(value))
    except (TypeError, ValueError):
        return False
