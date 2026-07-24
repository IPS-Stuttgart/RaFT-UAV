"""Compatibility validation for retention-aware Fortem track identifiers.

The maintained implementation lives in the sibling
``tracklet_viterbi_retention.py`` module. This package preserves the public
import path while ensuring malformed track identifiers cannot manufacture
support for an unrelated integer track.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "tracklet_viterbi_retention.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._tracklet_viterbi_retention_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        f"cannot load retention-aware tracklet-Viterbi implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_TRACK_SUPPORT_BY_ID = _IMPL._track_support_by_id


def _track_support_by_id(radar: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Return support only for exact integer Fortem track identifiers."""

    if radar.empty or "track_id" not in radar.columns:
        return _ORIGINAL_TRACK_SUPPORT_BY_ID(radar)
    track_ids = pd.Series(
        [_IMPL._base._optional_track_id(value) for value in radar["track_id"]],
        index=radar.index,
        dtype="Int64",
    )
    valid = track_ids.notna()
    if not bool(valid.any()):
        return {}
    normalized = radar.loc[valid].copy()
    normalized["track_id"] = track_ids.loc[valid].astype(int).to_numpy()
    return _ORIGINAL_TRACK_SUPPORT_BY_ID(normalized)


_IMPL._track_support_by_id = _track_support_by_id

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_ORIGINAL_TRACK_SUPPORT_BY_ID"] = _ORIGINAL_TRACK_SUPPORT_BY_ID
globals()["_track_support_by_id"] = _track_support_by_id

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
