"""Compatibility fixes for radar track-level feature extraction.

The maintained implementation lives in the sibling ``radar_track_features.py``
module. This package preserves the public import path while treating serialized
missing track identifiers as missing observations instead of shared tracks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


_IMPL_PATH = Path(__file__).resolve().parent.parent / "radar_track_features.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._radar_track_features_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load radar track features from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_ADD_TRACK_LEVEL_FEATURES = _IMPL.add_track_level_features
_MISSING_TRACK_ID_TEXT = frozenset({"", "nan", "none", "<na>"})


def _missing_track_id_mask(values: pd.Series) -> pd.Series:
    """Return rows whose track identifiers are absent or serialized missing values."""

    raw = pd.Series(values, index=values.index)
    missing = raw.isna()
    text = raw.astype("string").str.strip().str.lower()
    return missing | text.isin(_MISSING_TRACK_ID_TEXT).fillna(True)


def add_track_level_features(
    radar: pd.DataFrame,
    *,
    window_frames: int = 10,
) -> pd.DataFrame:
    """Append track features without sharing history across missing-like IDs."""

    if radar.empty or "track_id" not in radar.columns:
        return _ORIGINAL_ADD_TRACK_LEVEL_FEATURES(
            radar,
            window_frames=window_frames,
        )

    missing_track_ids = _missing_track_id_mask(radar["track_id"])
    if not bool(missing_track_ids.any()):
        return _ORIGINAL_ADD_TRACK_LEVEL_FEATURES(
            radar,
            window_frames=window_frames,
        )

    work = radar.copy()
    original_track_ids = radar["track_id"].to_numpy(copy=True)
    work.loc[missing_track_ids, "track_id"] = np.nan
    featured = _ORIGINAL_ADD_TRACK_LEVEL_FEATURES(
        work,
        window_frames=window_frames,
    )
    featured["track_id"] = original_track_ids
    return featured


_IMPL.add_track_level_features = add_track_level_features

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_ADD_TRACK_LEVEL_FEATURES"] = _ORIGINAL_ADD_TRACK_LEVEL_FEATURES
globals()["_MISSING_TRACK_ID_TEXT"] = _MISSING_TRACK_ID_TEXT
globals()["_missing_track_id_mask"] = _missing_track_id_mask
globals()["add_track_level_features"] = add_track_level_features

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
