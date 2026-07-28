"""Compatibility fixes for tracklet feature-store diagnostics.

The maintained implementation lives in the sibling ``tracklet_feature_store.py``
module. This package preserves the public import path while matching external
selected-radar rows by stable opaque track ID, with track-index fallback only when
an ID is unavailable, and parsing persisted Boolean diagnostics explicitly instead
of relying on Python string truthiness.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_feature_store.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._tracklet_feature_store_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load tracklet feature-store implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_COUNTERFACTUAL_ASSOCIATION_DASHBOARD = (
    _IMPL.build_counterfactual_association_dashboard
)
_ORIGINAL_SUMMARIZE_COUNTERFACTUAL_REGRET = _IMPL.summarize_counterfactual_regret
_TRUE_BOOLEAN_TEXT = frozenset({"true", "t", "yes", "y", "on"})
_FALSE_BOOLEAN_TEXT = frozenset(
    {"false", "f", "no", "n", "off", "", "nan", "none", "null", "<na>", "nat"}
)
_MISSING_IDENTIFIER_TEXT = frozenset({"", "nan", "none", "null", "<na>", "nat"})


def _stable_identifier(value: object) -> object | None:
    """Return a hashable identifier without discarding opaque or fractional IDs."""

    if value is None:
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return None
    if isinstance(value, (bool, np.bool_)):
        return ("bool", bool(value))
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, int):
        return ("number", int(value))
    if isinstance(value, float):
        number = float(value)
        if not np.isfinite(number):
            return None
        return ("number", int(number) if number.is_integer() else number)

    text = str(value).strip()
    if text.casefold() in _MISSING_IDENTIFIER_TEXT:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return ("text", text)
    if not np.isfinite(number):
        return None
    return ("number", int(number) if number.is_integer() else number)


def _identifier_key(
    row: pd.Series,
    identifier: str,
    value: object,
) -> tuple[object, ...]:
    """Return one frame-scoped candidate identifier key."""

    return (
        row.get("frame_key_type"),
        row.get("frame_key"),
        identifier,
        value,
    )


def _candidate_match_key(row: pd.Series) -> tuple[object, ...] | None:
    """Return the strongest available frame-scoped candidate identity."""

    track_id = _stable_identifier(row.get("track_id"))
    if track_id is not None:
        return _identifier_key(row, "track_id", track_id)
    track_index = _stable_identifier(row.get("track_index"))
    if track_index is not None:
        return _identifier_key(row, "track_index", track_index)
    return None


def _selection_mask(
    features: pd.DataFrame,
    selected_radar: pd.DataFrame | None,
) -> np.ndarray:
    """Match selected candidates without requiring unstable track-index parity."""

    if selected_radar is None or selected_radar.empty:
        return np.zeros(len(features), dtype=bool)

    selected = _IMPL._append_frame_keys(selected_radar)
    selected_track_ids: set[tuple[object, ...]] = set()
    selected_track_indices: set[tuple[object, ...]] = set()
    fallback_track_indices: set[tuple[object, ...]] = set()
    for _, row in selected.iterrows():
        track_id = _stable_identifier(row.get("track_id"))
        track_index = _stable_identifier(row.get("track_index"))
        if track_id is not None:
            selected_track_ids.add(_identifier_key(row, "track_id", track_id))
        if track_index is not None:
            index_key = _identifier_key(row, "track_index", track_index)
            selected_track_indices.add(index_key)
            if track_id is None:
                fallback_track_indices.add(index_key)

    def is_selected(row: pd.Series) -> bool:
        track_id = _stable_identifier(row.get("track_id"))
        track_index = _stable_identifier(row.get("track_index"))
        if track_id is not None:
            id_key = _identifier_key(row, "track_id", track_id)
            if id_key in selected_track_ids:
                return True
        if track_index is None:
            return False
        index_key = _identifier_key(row, "track_index", track_index)
        if track_id is None:
            return index_key in selected_track_indices
        return index_key in fallback_track_indices

    return np.fromiter(
        (is_selected(row) for _, row in features.iterrows()),
        dtype=bool,
        count=len(features),
    )


def _boolean_series(values: Any, *, column: str) -> pd.Series:
    """Parse native and serialized Boolean diagnostics without string truthiness."""

    series = pd.Series(values, copy=False)
    if series.empty:
        return pd.Series(index=series.index, dtype=bool)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean").fillna(False).astype(bool)

    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype("string").str.strip().str.casefold()
    truthy = (numeric.eq(1.0) | text.isin(_TRUE_BOOLEAN_TEXT)).fillna(False)
    falsy = (
        series.isna() | numeric.eq(0.0) | text.isin(_FALSE_BOOLEAN_TEXT)
    ).fillna(False)
    invalid = ~(truthy | falsy)
    if bool(invalid.any()):
        indices = series.index[invalid].tolist()
        invalid_values = series.loc[indices].tolist()
        raise ValueError(
            f"{column} contains invalid Boolean values at rows {indices}: "
            f"{invalid_values}"
        )
    return truthy.astype(bool)


def build_counterfactual_association_dashboard(features: Any) -> pd.DataFrame:
    """Build the dashboard after normalizing persisted selection flags."""

    normalized = pd.DataFrame(features).copy().reset_index(drop=True)
    column = "chosen_by_selected_radar"
    if column in normalized.columns:
        normalized[column] = _boolean_series(normalized[column], column=column)
    return _ORIGINAL_BUILD_COUNTERFACTUAL_ASSOCIATION_DASHBOARD(normalized)


def summarize_counterfactual_regret(regret: Any) -> dict[str, Any]:
    """Summarize regret after normalizing persisted availability flags."""

    normalized = pd.DataFrame(regret).copy()
    for column in ("truth_available", "selected_present"):
        if column in normalized.columns:
            normalized[column] = _boolean_series(normalized[column], column=column)
    return _ORIGINAL_SUMMARIZE_COUNTERFACTUAL_REGRET(normalized)


_IMPL._candidate_match_key = _candidate_match_key
_IMPL._selection_mask = _selection_mask
_IMPL.build_counterfactual_association_dashboard = (
    build_counterfactual_association_dashboard
)
_IMPL.summarize_counterfactual_regret = summarize_counterfactual_regret

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_stable_identifier"] = _stable_identifier
globals()["_identifier_key"] = _identifier_key
globals()["_candidate_match_key"] = _candidate_match_key
globals()["_selection_mask"] = _selection_mask
globals()["_boolean_series"] = _boolean_series
globals()["build_counterfactual_association_dashboard"] = (
    build_counterfactual_association_dashboard
)
globals()["summarize_counterfactual_regret"] = summarize_counterfactual_regret

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
