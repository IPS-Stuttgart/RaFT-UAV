"""Compatibility wrapper for reusable gap-threshold iterables.

The implementation remains in the sibling module. This package wrapper preserves
its public API while ensuring one-shot iterables passed to ``summarize_frame_gap``
are reused for every group and oracle column.
"""
from __future__ import annotations

from collections.abc import Iterable as _Iterable
import importlib.util
from pathlib import Path
import sys

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_reservoir_mixture_gap_frames.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_reservoir_mixture_gap_frames_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy gap-frame implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_SUMMARIZE_FRAME_GAP = _IMPL.summarize_frame_gap

for _name, _value in vars(_IMPL).items():
    if not _name.startswith("__"):
        globals()[_name] = _value


def summarize_frame_gap(
    frame_gap: pd.DataFrame,
    *,
    group_column: str | None = None,
    gap_thresholds_m: _Iterable[float] = _IMPL.DEFAULT_GAP_THRESHOLDS_M,
) -> pd.DataFrame:
    """Build summaries while reusing thresholds across all groups and oracles."""

    thresholds = tuple(gap_thresholds_m)
    return _ORIGINAL_SUMMARIZE_FRAME_GAP(
        frame_gap,
        group_column=group_column,
        gap_thresholds_m=thresholds,
    )


_IMPL.summarize_frame_gap = summarize_frame_gap
globals()["summarize_frame_gap"] = summarize_frame_gap

__all__ = sorted(name for name in vars(_IMPL) if not name.startswith("_"))
