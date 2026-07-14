"""Compatibility package with deterministic greedy-MOT tie breaking.

The maintained implementation lives in the sibling ``mot.py`` module. This
package preserves the public import path while making exact association-distance
ties independent of set iteration order.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_IMPL_PATH = Path(__file__).resolve().parent.parent / "mot.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._mot_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load MMUAD MOT implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _nearest_track_id(
    z: np.ndarray,
    active: dict[int, Any],
    unmatched_tracks: set[int],
    config: Any,
) -> int | None:
    """Return the nearest eligible track with a stable lowest-ID tie break."""

    if not np.isfinite(z).all():
        return None
    best_track: int | None = None
    best_distance = float("inf")
    for track_id in sorted(unmatched_tracks):
        distance = float(np.linalg.norm(active[track_id].state[:3] - z))
        if distance < best_distance:
            best_distance = distance
            best_track = track_id
    if best_distance <= config.max_association_distance_m:
        return best_track
    return None


_IMPL._nearest_track_id = _nearest_track_id

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_nearest_track_id"] = _nearest_track_id

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
