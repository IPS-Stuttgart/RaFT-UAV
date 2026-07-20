"""Compatibility fixes for paper-style radar preselection.

The maintained implementation lives in the sibling ``paper_selection.py``
module. This package preserves the public import path while excluding non-finite
class probabilities and preserving exact integer-like track identifiers in
selection metadata and segment tie-breaking.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "paper_selection.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav._paper_selection_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load paper-selection implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)


def _catprob_candidate_pool(
    candidates: pd.DataFrame,
    catprob_threshold: float | None,
) -> pd.DataFrame:
    """Apply the class-probability gate without accepting NaN or infinity."""

    if catprob_threshold is None or "cat_prob_uav" not in candidates.columns:
        return candidates.copy()
    catprob = pd.to_numeric(
        candidates["cat_prob_uav"],
        errors="coerce",
    ).to_numpy(dtype=float)
    pool = candidates.loc[
        np.isfinite(catprob) & (catprob >= float(catprob_threshold))
    ].copy()
    if not pool.empty:
        pool["association_catprob_threshold"] = float(catprob_threshold)
        pool["association_catprob_candidate_rows"] = int(len(candidates))
    return pool


def _mean_catprob(frame: pd.DataFrame) -> float:
    """Return the mean of finite class probabilities for track tie-breaking."""

    if "cat_prob_uav" not in frame.columns or frame.empty:
        return 0.0
    catprob = pd.to_numeric(
        frame["cat_prob_uav"],
        errors="coerce",
    ).to_numpy(dtype=float)
    finite = catprob[np.isfinite(catprob)]
    return float(np.mean(finite)) if finite.size else 0.0


def _track_id_from_frame(frame: pd.DataFrame) -> int:
    """Return the first exact integer-like track ID without float round-trips."""

    if "track_id" not in frame.columns:
        return -1
    for value in frame["track_id"].tolist():
        track_id = optional_int(value)
        if track_id is not None:
            return track_id
    return -1


_LEGACY._catprob_candidate_pool = _catprob_candidate_pool
_LEGACY._mean_catprob = _mean_catprob
_LEGACY._track_id_from_frame = _track_id_from_frame

globals().update(
    {
        name: getattr(_LEGACY, name)
        for name in dir(_LEGACY)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_catprob_candidate_pool"] = _catprob_candidate_pool
globals()["_mean_catprob"] = _mean_catprob
globals()["_track_id_from_frame"] = _track_id_from_frame

__doc__ = _LEGACY.__doc__
__all__ = [
    name for name in dir(_LEGACY) if not (name.startswith("__") and name.endswith("__"))
]
