"""Compatibility fixes for fifth-wave reliability diagnostics.

The maintained implementation lives in the sibling ``fifth_wave_diagnostics.py``
module. This package preserves the public import path while keeping serialized
track identifiers exact in purity summaries.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from raft_uav.numeric import optional_int

_IMPL_PATH = Path(__file__).resolve().parent.parent / "fifth_wave_diagnostics.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.evaluation._fifth_wave_diagnostics_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load fifth-wave diagnostics from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def track_purity_summary(
    selected_radar: pd.DataFrame,
    *,
    track_column: str = "track_id",
) -> dict[str, float | int | None]:
    """Return track purity without truncating or rounding track identifiers."""

    if selected_radar.empty or track_column not in selected_radar.columns:
        return {
            "selected_radar_rows": int(len(selected_radar)),
            "dominant_track_id": None,
            "dominant_track_fraction": np.nan,
            "selected_track_entropy": np.nan,
            "selected_track_count": 0,
        }

    track_ids = [
        track_id
        for value in selected_radar[track_column].tolist()
        if (track_id := optional_int(value)) is not None
    ]
    if not track_ids:
        return {
            "selected_radar_rows": int(len(selected_radar)),
            "dominant_track_id": None,
            "dominant_track_fraction": np.nan,
            "selected_track_entropy": np.nan,
            "selected_track_count": 0,
        }

    counts = pd.Series(track_ids, dtype=object).value_counts(sort=True)
    probabilities = counts.to_numpy(dtype=float) / float(counts.sum())
    return {
        "selected_radar_rows": int(len(selected_radar)),
        "dominant_track_id": int(counts.index[0]),
        "dominant_track_fraction": float(probabilities[0]),
        "selected_track_entropy": _IMPL._entropy(probabilities),
        "selected_track_count": int(len(counts)),
    }


_IMPL.track_purity_summary = track_purity_summary

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["track_purity_summary"] = track_purity_summary

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
