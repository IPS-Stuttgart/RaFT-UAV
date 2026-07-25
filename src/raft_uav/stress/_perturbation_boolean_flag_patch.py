"""Normalize persisted false-track flags before stress perturbation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from . import perturbations as _IMPL

_ORIGINAL_INJECT_FALSE_TRACKS = _IMPL.inject_false_tracks
_TRUE_BOOLEAN_TEXT = frozenset({"true", "t", "yes", "y", "1", "1.0"})
_FALSE_BOOLEAN_TEXT = frozenset(
    {
        "false",
        "f",
        "no",
        "n",
        "0",
        "0.0",
        "",
        "nan",
        "none",
        "null",
        "<na>",
        "nat",
    }
)


def _boolean_series(values: Any, *, column: str) -> pd.Series:
    """Parse native and serialized Boolean flags without string truthiness."""

    series = pd.Series(values, copy=False)
    if series.empty:
        return pd.Series(index=series.index, dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    truthy = (text.isin(_TRUE_BOOLEAN_TEXT) | numeric.eq(1.0)).fillna(False)
    falsy = (
        series.isna() | text.isin(_FALSE_BOOLEAN_TEXT) | numeric.eq(0.0)
    ).fillna(False)
    invalid = ~(truthy | falsy)
    if bool(invalid.any()):
        invalid_indices = invalid[invalid].index.tolist()
        invalid_values = series.loc[invalid_indices].tolist()
        raise ValueError(
            f"{column} contains invalid Boolean values at rows "
            f"{invalid_indices}: {invalid_values}"
        )
    return truthy.astype(bool)


def inject_false_tracks(
    frame: pd.DataFrame,
    *,
    false_tracks_per_frame: int,
    position_std_m: float,
    rng: Any,
) -> pd.DataFrame:
    """Inject false tracks after normalizing any persisted diagnostic flags."""

    normalized = frame.copy()
    column = "stress_false_track"
    if column in normalized.columns:
        normalized[column] = _boolean_series(normalized[column], column=column)
    return _ORIGINAL_INJECT_FALSE_TRACKS(
        normalized,
        false_tracks_per_frame=false_tracks_per_frame,
        position_std_m=position_std_m,
        rng=rng,
    )


_IMPL.inject_false_tracks = inject_false_tracks
