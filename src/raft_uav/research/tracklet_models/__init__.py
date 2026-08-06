"""Compatibility wrapper keeping research aggregations sequence-local."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


def tracklet_feature_frame(
    radar: pd.DataFrame,
    *,
    max_frame_gap: float = 1.5,
) -> pd.DataFrame:
    """Aggregate tracklets independently for every explicit sequence."""

    if radar.empty or "sequence_id" not in radar.columns:
        return _ORIGINAL_TRACKLET_FEATURE_FRAME(radar, max_frame_gap=max_frame_gap)

    feature_frames: list[pd.DataFrame] = []
    for sequence_id, sequence_rows in radar.groupby(
        "sequence_id",
        sort=False,
        dropna=False,
    ):
        features = _ORIGINAL_TRACKLET_FEATURE_FRAME(
            sequence_rows,
            max_frame_gap=max_frame_gap,
        )
        if features.empty:
            continue
        features.insert(0, "sequence_id", sequence_id)
        feature_frames.append(features)
    if not feature_frames:
        return pd.DataFrame()
    return pd.concat(feature_frames, ignore_index=True)


def estimate_frame_clutter_density(radar: pd.DataFrame) -> dict[str, float]:
    """Estimate clutter over sequence-local physical radar frames."""

    if radar.empty:
        return {
            "mean_candidates_per_frame": 0.0,
            "p95_candidates_per_frame": 0.0,
        }
    group_key = "frame_index" if "frame_index" in radar.columns else "time_s"
    group_columns = [group_key]
    if "sequence_id" in radar.columns:
        group_columns.insert(0, "sequence_id")
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
        probs = (
            pd.to_numeric(radar["cat_prob_uav"], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if probs.size:
            out["mean_cat_prob_uav"] = float(np.mean(probs))
            out["low_cat_prob_rate"] = float(np.mean(probs < 0.4))
    return out


_IMPL.tracklet_feature_frame = tracklet_feature_frame
_IMPL.estimate_frame_clutter_density = estimate_frame_clutter_density

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["tracklet_feature_frame"] = tracklet_feature_frame
globals()["estimate_frame_clutter_density"] = estimate_frame_clutter_density

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
