"""Compatibility fixes for assignment-block identifiers and flight scoping.

The maintained implementation lives in the sibling
``candidate_assignment_blocks.py`` module. This package preserves the public
import path while filtering genuinely missing sequence identifiers and preventing
independent physical flights from being merged into synthetic failure blocks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from pandas.api.types import is_scalar

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
_SCOPE_TOKEN_PREFIX = "__raft_uav_assignment_block_scope_"
_MISSING_SCOPE_TEXT = frozenset({"", "nan", "none", "<na>", "nat"})


def _drop_missing_sequence_ids(frame_rows: Any) -> pd.DataFrame:
    """Remove rows whose sequence identifier is genuinely missing."""

    rows = pd.DataFrame(frame_rows).copy()
    if rows.empty or "sequence_id" not in rows.columns:
        return rows
    return rows.loc[rows["sequence_id"].notna()].copy()


def _canonical_flight_id(value: object) -> str | None:
    """Return one normalized scalar flight identifier or ``None`` when missing."""

    if not is_scalar(value):
        raise ValueError("flight_id values must be scalar")
    if value is None:
        return None
    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False
    if missing:
        return None
    text = str(value).strip()
    return None if text.casefold() in _MISSING_SCOPE_TEXT else text


def _scope_rows_by_flight(
    frame_rows: Any,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str | None]] | None]:
    """Replace sequence IDs with joint sequence/flight tokens when available."""

    rows = _drop_missing_sequence_ids(frame_rows)
    if rows.empty or "flight_id" not in rows.columns or "sequence_id" not in rows.columns:
        return rows, None

    tokens: dict[tuple[str, str | None], str] = {}
    metadata: dict[str, tuple[str, str | None]] = {}
    row_tokens: list[str] = []
    for sequence_id, flight_id in zip(
        rows["sequence_id"].tolist(),
        rows["flight_id"].tolist(),
        strict=True,
    ):
        scope = (str(sequence_id), _canonical_flight_id(flight_id))
        token = tokens.get(scope)
        if token is None:
            token = f"{_SCOPE_TOKEN_PREFIX}{len(tokens)}"
            tokens[scope] = token
            metadata[token] = scope
        row_tokens.append(token)
    rows["sequence_id"] = row_tokens
    return rows, metadata


def _restore_scope_columns(
    rows: pd.DataFrame,
    metadata: dict[str, tuple[str, str | None]] | None,
) -> pd.DataFrame:
    """Restore public sequence and flight identifiers after temporary scoping."""

    frame = pd.DataFrame(rows).copy()
    if metadata is None or frame.empty or "sequence_id" not in frame.columns:
        return frame

    restored_sequence: list[str] = []
    restored_flight: list[object] = []
    for value in frame["sequence_id"].astype(str).tolist():
        if value == "__pooled__":
            restored_sequence.append(value)
            restored_flight.append("__pooled__")
            continue
        scope = metadata.get(value)
        if scope is None:  # pragma: no cover - internal contract guard
            raise RuntimeError(
                f"assignment blocks returned unknown internal scope token {value!r}"
            )
        sequence_id, flight_id = scope
        restored_sequence.append(sequence_id)
        restored_flight.append(pd.NA if flight_id is None else flight_id)
    frame["sequence_id"] = restored_sequence
    frame["flight_id"] = restored_flight
    return frame


def build_candidate_assignment_block_tables(
    frame_rows: pd.DataFrame,
    *,
    max_gap_s: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build assignment blocks independently within each physical flight."""

    scoped_rows, metadata = _scope_rows_by_flight(frame_rows)
    blocks, summary = _ORIGINAL_BUILD(
        scoped_rows,
        max_gap_s=max_gap_s,
    )
    return (
        _restore_scope_columns(blocks, metadata),
        _restore_scope_columns(summary, metadata),
    )


_IMPL._drop_missing_sequence_ids = _drop_missing_sequence_ids
_IMPL._canonical_flight_id = _canonical_flight_id
_IMPL._scope_rows_by_flight = _scope_rows_by_flight
_IMPL._restore_scope_columns = _restore_scope_columns
_IMPL.build_candidate_assignment_block_tables = build_candidate_assignment_block_tables

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_drop_missing_sequence_ids"] = _drop_missing_sequence_ids
globals()["_canonical_flight_id"] = _canonical_flight_id
globals()["_scope_rows_by_flight"] = _scope_rows_by_flight
globals()["_restore_scope_columns"] = _restore_scope_columns
globals()["build_candidate_assignment_block_tables"] = (
    build_candidate_assignment_block_tables
)

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
