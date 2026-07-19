"""Compatibility wrapper preserving exact Fortem track identifiers.

The maintained implementation lives in the sibling ``topk_weakz_tracklet.py``
module. This package keeps the public import path while preventing large,
fractional, Boolean, or malformed track identifiers from being rounded or
silently merged when tracklets are grouped.
"""

from __future__ import annotations

from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int as _optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "topk_weakz_tracklet.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.baselines._topk_weakz_tracklet_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load top-k weak-z tracklet implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_BUILD_FORTEM_TRACKLETS = _IMPL.build_fortem_tracklets


def build_fortem_tracklets(
    radar: pd.DataFrame,
    config: Any = None,
):
    """Build tracklets without routing integer identifiers through ``float``.

    The legacy implementation uses floating-point group keys. Dense surrogate
    integers preserve its grouping behavior while exact identifiers are restored
    on the returned immutable tracklet records. Invalid identifiers remain
    missing and therefore retain the legacy row-isolation behavior.
    """

    frame = pd.DataFrame(radar).copy()
    if frame.empty or "track_id" not in frame.columns:
        return _ORIGINAL_BUILD_FORTEM_TRACKLETS(frame, config)

    exact_ids = [_optional_int(value) for value in frame["track_id"].tolist()]
    surrogate_by_id: dict[int, int] = {}
    surrogates: list[float | int] = []
    for exact_id in exact_ids:
        if exact_id is None:
            surrogates.append(np.nan)
            continue
        surrogate = surrogate_by_id.setdefault(exact_id, len(surrogate_by_id))
        surrogates.append(surrogate)
    frame["track_id"] = surrogates

    tracklets = _ORIGINAL_BUILD_FORTEM_TRACKLETS(frame, config)
    restored = []
    for tracklet in tracklets:
        first_row = tracklet.row_indices[0]
        restored.append(replace(tracklet, track_id=exact_ids[first_row]))
    return restored


_IMPL._optional_int = _optional_int
_IMPL.build_fortem_tracklets = build_fortem_tracklets

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_optional_int"] = _optional_int
globals()["build_fortem_tracklets"] = build_fortem_tracklets
__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
