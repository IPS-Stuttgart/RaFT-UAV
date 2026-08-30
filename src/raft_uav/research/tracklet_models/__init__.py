"""Compatibility wrapper keeping research aggregations flight/sequence-local."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_models.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.research._tracklet_models_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load tracklet-model implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACKLET_FEATURE_FRAME = _IMPL.tracklet_feature_frame
_SCOPE_COLUMN_ALIASES = ("sequence_id", "flight_id")


def _validated_max_frame_gap(value: Any) -> float:
    """Return a finite non-negative scalar tracklet segmentation threshold."""

    message = "max_frame_gap must be a finite non-negative real scalar"
    current = value
    seen: set[int] = set()
    while True:
        if isinstance(current, (bool, np.bool_)) or np.ma.is_masked(current):
            raise ValueError(message)
        if isinstance(current, (complex, np.complexfloating)):
            raise ValueError(message)
        try:
            scalar = np.asarray(current)
        except (TypeError, ValueError) as exc:
            raise ValueError(message) from exc
        if scalar.ndim != 0 or scalar.dtype.kind in {"b", "c"}:
            raise ValueError(message)
        if scalar.dtype.kind != "O":
            current = scalar.item()
            break
        marker = id(current)
        if marker in seen:
            raise ValueError(message)
        seen.add(marker)
        item = scalar.item()
        if item is current:
            raise ValueError(message)
        current = item

    try:
        gap = float(current)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError(message)
    return gap


def _scope_columns(radar: pd.DataFrame) -> list[str]:
    """Return every explicit flight/sequence identifier available on ``radar``."""

    return [column for column in _SCOPE_COLUMN_ALIASES if column in radar.columns]


def tracklet_feature_frame(
    radar: pd.DataFrame,
    *,
    max_frame_gap: float = 1.5,
) -> pd.DataFrame:
    """Aggregate tracklets independently for every explicit flight/sequence."""

    normalized_max_frame_gap = _validated_max_frame_gap(max_frame_gap)
    scope_columns = _scope_columns(radar)
    if radar.empty or not scope_columns:
        return _ORIGINAL_TRACKLET_FEATURE_FRAME(
            radar,
            max_frame_gap=normalized_max_frame_gap,
        )

    grouper: str | list[str] = scope_columns[0] if len(scope_columns) == 1 else scope_columns
    feature_frames: list[pd.DataFrame] = []
    for scope_key, scoped_rows in radar.groupby(
        grouper,
        sort=False,
        dropna=False,
    ):
        features = _ORIGINAL_TRACKLET_FEATURE_FRAME(
            scoped_rows,
            max_frame_gap=normalized_max_frame_gap,
        )
        if features.empty:
            continue
        scope_values = (scope_key,) if len(scope_columns) == 1 else tuple(scope_key)
        for column, value in reversed(list(zip(scope_columns, scope_values, strict=True))):
            features.insert(0, column, value)
        feature_frames.append(features)
    if not feature_frames:
        return pd.DataFrame()
    return pd.concat(feature_frames, ignore_index=True)


def estimate_frame_clutter_density(radar: pd.DataFrame) -> dict[str, float]:
    """Estimate clutter over flight/sequence-local physical radar frames."""

    if radar.empty:
        return {
            "mean_candidates_per_frame": 0.0,
            "p95_candidates_per_frame": 0.0,
        }
    group_key = "frame_index" if "frame_index" in radar.columns else "time_s"
    group_columns = [*_scope_columns(radar), group_key]
    counts = (
        radar.groupby(group_columns, sort=False, dropna=False)
        .size()
        .to_numpy(dtype=float)
    )
    out = {
        "mean_candidates_per_frame": float(np.mean(counts)),
        "p95_candidates_per_frame": float(np.percentile(counts, 95)),
    }
    if "cat_prob_uav" in radar.columns:
        probs = pd.to_numeric(radar["cat_prob_uav"], errors="coerce").to_numpy(dtype=float)
        probs = probs[np.isfinite(probs)]
        if probs.size:
            out["mean_cat_prob_uav"] = float(np.mean(probs))
            out["low_cat_prob_rate"] = float(np.mean(probs < 0.4))
    return out


_IMPL._validated_max_frame_gap = _validated_max_frame_gap
_IMPL._scope_columns = _scope_columns
_IMPL.tracklet_feature_frame = tracklet_feature_frame
_IMPL.estimate_frame_clutter_density = estimate_frame_clutter_density

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_validated_max_frame_gap"] = _validated_max_frame_gap
globals()["_scope_columns"] = _scope_columns
globals()["tracklet_feature_frame"] = tracklet_feature_frame
globals()["estimate_frame_clutter_density"] = estimate_frame_clutter_density

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
