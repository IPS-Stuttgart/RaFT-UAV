"""Compatibility guard for Track 5 scorecard per-sequence comparisons.

The maintained implementation lives in the sibling
``track5_scorecard_compare.py`` module. This package preserves the public import
path while rejecting duplicate per-sequence rows instead of silently keeping the
first occurrence.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_scorecard_compare.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_scorecard_compare_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 scorecard comparison implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_NORMALIZE_POSE_TABLE = _IMPL._normalize_pose_by_sequence_table


def _normalize_pose_by_sequence_table(rows: Any, *, label: str):
    """Reject duplicate sequence rows instead of silently keeping the first one."""

    frame = _IMPL.pd.DataFrame(rows).copy()
    sequence_column = None
    if "sequence_id" in frame.columns:
        sequence_column = "sequence_id"
    elif "sequence" in frame.columns:
        sequence_column = "sequence"

    if sequence_column is not None:
        sequence_ids = frame[sequence_column].astype(str)
        duplicate = sequence_ids.duplicated(keep=False)
        if duplicate.any():
            examples = sequence_ids.loc[duplicate].drop_duplicates().head(5).tolist()
            raise ValueError(
                f"{label} pose table contains duplicate sequence_id rows: {examples}"
            )

    return _ORIGINAL_NORMALIZE_POSE_TABLE(frame, label=label)


_IMPL._normalize_pose_by_sequence_table = _normalize_pose_by_sequence_table

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_normalize_pose_by_sequence_table"] = _normalize_pose_by_sequence_table

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
