"""Identifier normalization for MMUAD temporal candidate models.

Candidate CSVs can represent the same numeric tracker identifier with different
scalar types, for example ``491``, ``491.0``, and ``"491.0"``.  Temporal priors
must treat those values as the same track without collapsing deliberately
opaque identifiers such as ``"001"`` into integer ``1``.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import numbers
import re
from typing import Any, Iterable

import numpy as np
import pandas as pd

_INVALID_TRACK_IDS = {
    "",
    "nan",
    "none",
    "null",
    "<na>",
    "-1",
    "-1.0",
    "false",
    "true",
}
_CANONICAL_INTEGER_TEXT = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)(?:\.0+)?$")


def _exact_rational_text(numerator: int, denominator: int) -> str:
    """Return an exact decimal when finite, otherwise a reduced fraction."""

    if denominator <= 0:
        raise ValueError("rational denominator must be positive")
    factor_two = 0
    factor_five = 0
    remainder = denominator
    while remainder % 2 == 0:
        factor_two += 1
        remainder //= 2
    while remainder % 5 == 0:
        factor_five += 1
        remainder //= 5
    if remainder != 1:
        return f"{numerator}/{denominator}"

    decimal_places = max(factor_two, factor_five)
    scaled = abs(numerator)
    scaled *= 2 ** (decimal_places - factor_two)
    scaled *= 5 ** (decimal_places - factor_five)
    digits = str(scaled).rjust(decimal_places + 1, "0")
    if decimal_places:
        digits = f"{digits[:-decimal_places]}.{digits[-decimal_places:]}"
        digits = digits.rstrip("0").rstrip(".")
    sign = "-" if numerator < 0 else ""
    return f"{sign}{digits}"


def canonical_track_id(value: Any) -> str | None:
    """Return a stable track identity while preserving opaque leading zeros.

    Integral numeric scalars and canonical integer strings are normalized to an
    integer string.  Strings with leading zeros remain opaque, so ``"001"`` is
    not treated as the same identifier as numeric ``1``. Boolean-like values are
    treated as missing because schema coercion must not turn them into shared
    temporal identities.
    """

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, numbers.Integral):
        number = int(value)
        return None if number == -1 else str(number)
    if isinstance(value, numbers.Rational):
        numerator = int(value.numerator)
        denominator = int(value.denominator)
        if denominator == 1:
            return None if numerator == -1 else str(numerator)
        return _exact_rational_text(numerator, denominator)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number.is_integer():
            integer = int(number)
            return None if integer == -1 else str(integer)
        return format(number, ".17g")

    text = str(value).strip()
    if text.lower() in _INVALID_TRACK_IDS:
        return None
    if not _CANONICAL_INTEGER_TEXT.fullmatch(text):
        return text
    try:
        integer = int(Decimal(text))
    except (InvalidOperation, ValueError, OverflowError):
        return text
    return None if integer == -1 else str(integer)


def canonical_track_ids(values: Iterable[Any]) -> np.ndarray:
    """Vectorize :func:`canonical_track_id` into an object array."""

    return np.asarray([canonical_track_id(value) for value in values], dtype=object)
