"""CSV readers for MMUAD class probability tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SEQUENCE_ALIASES = (
    "sequence_id",
    "Sequence",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "clip",
    "clip_id",
)


def read_sequence_text_csv(path: Path) -> pd.DataFrame:
    """Read CSV input while preserving opaque sequence ids as text."""

    try:
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    except TypeError:
        rows = pd.read_csv(path)
    out = rows.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return _canonicalize_sequence_id_column(out)


def read_class_probability_csv(path: Path) -> pd.DataFrame:
    """Read classifier CSV output while preserving sequence ids as text."""

    return read_sequence_text_csv(path)


def _canonicalize_sequence_id_column(rows: pd.DataFrame) -> pd.DataFrame:
    """Add ``sequence_id`` when the input uses a supported sequence alias."""

    out = rows.copy()
    if "sequence_id" in out.columns:
        out["sequence_id"] = out["sequence_id"].astype(str)
        return out

    lower_to_column = {str(column).strip().lower(): column for column in out.columns}
    source_column = None
    for alias in SEQUENCE_ALIASES:
        source_column = lower_to_column.get(alias.lower())
        if source_column is not None:
            break
    if source_column is None:
        return out

    out.insert(0, "sequence_id", out[source_column].astype(str))
    return out
