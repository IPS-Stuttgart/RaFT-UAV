"""Reject complex pseudo-Booleans in oracle-coverage summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import oracle_coverage as _oracle_coverage

_ORIGINAL_SERIALIZED_BOOLEAN_SERIES = _oracle_coverage._serialized_boolean_series


def _is_complex_scalar(value: Any) -> bool:
    """Return whether scalar array wrappers contain a complex value."""

    scalar = value
    seen_arrays: set[int] = set()
    while isinstance(scalar, np.ndarray):
        if scalar.ndim != 0:
            return False
        array_id = id(scalar)
        if array_id in seen_arrays:
            return False
        seen_arrays.add(array_id)
        scalar = scalar.item()
    if isinstance(scalar, np.generic):
        scalar = scalar.item()
    return isinstance(scalar, complex)


def _serialized_boolean_series(values: Any, *, column: str) -> pd.Series:
    """Normalize Boolean diagnostics after rejecting complex pseudo-values."""

    series = pd.Series(values)
    complex_values = series.map(_is_complex_scalar).fillna(False).astype(bool)
    if bool(complex_values.any()):
        invalid_indices = complex_values[complex_values].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        raise ValueError(
            f"{column} contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return _ORIGINAL_SERIALIZED_BOOLEAN_SERIES(series, column=column)


_oracle_coverage._serialized_boolean_series = _serialized_boolean_series
