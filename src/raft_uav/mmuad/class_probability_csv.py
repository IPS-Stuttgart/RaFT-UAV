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
        rows = pd.read_csv(path, dtype=str)
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
        out["sequence_id"] = _sequence_id_text(out["sequence_id"])
        return out

    lower_to_column = {str(column).strip().lower(): column for column in out.columns}
    source_column = None
    for alias in SEQUENCE_ALIASES:
        source_column = lower_to_column.get(alias.lower())
        if source_column is not None:
            break
    if source_column is None:
        return out

    out.insert(0, "sequence_id", _sequence_id_text(out[source_column]))
    return out


def _sequence_id_text(values: pd.Series) -> pd.Series:
    """Return stripped sequence ids without changing opaque text such as ``001``."""

    return values.where(values.notna(), "").astype(str).str.strip()
