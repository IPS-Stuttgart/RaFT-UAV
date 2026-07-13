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
    if isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number):
            return None
        if number.is_integer():
            integer = int(number)
            return None if integer == -1 else str(integer)
        return str(value)

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
