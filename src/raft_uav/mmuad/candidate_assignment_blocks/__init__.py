"""Compatibility package excluding missing assignment-block sequence identifiers.

The maintained implementation lives in the sibling
``candidate_assignment_blocks.py`` module. This package preserves the public
import path while filtering genuinely missing sequence identifiers before the
legacy implementation converts identifiers to strings.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "candidate_assignment_blocks.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_assignment_blocks_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load candidate assignment blocks from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_BUILD = _IMPL.build_candidate_assignment_block_tables


def _drop_missing_sequence_ids(frame_rows: Any) -> pd.DataFrame:
    """Remove rows whose sequence identifier is genuinely missing."""

    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty or "sequence_id" not in rows.columns:
        return rows
    return rows.loc[rows["sequence_id"].notna()].copy()


def build_candidate_assignment_block_tables(
    frame_rows: pd.DataFrame,
    *,
    max_gap_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build assignment blocks without materializing missing IDs as strings."""

    return _ORIGINAL_BUILD(
        _drop_missing_sequence_ids(frame_rows),
        max_gap_s=max_gap_s,
    )


_IMPL._drop_missing_sequence_ids = _drop_missing_sequence_ids
_IMPL.build_candidate_assignment_block_tables = build_candidate_assignment_block_tables

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_drop_missing_sequence_ids"] = _drop_missing_sequence_ids
globals()["build_candidate_assignment_block_tables"] = (
    build_candidate_assignment_block_tables
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
