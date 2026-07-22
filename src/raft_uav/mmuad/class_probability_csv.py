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
        rows = pd.read_csv(path, dtype=str, na_filter=False)
    out = rows.copy()
    normalized_columns = [str(column).strip() for column in out.columns]
    duplicated = pd.Index(normalized_columns).duplicated(keep=False)
    duplicate_columns = sorted(
        {
            column
            for column, is_duplicate in zip(normalized_columns, duplicated)
            if is_duplicate
        }
    )
    if duplicate_columns:
        duplicate_text = ", ".join(repr(column) for column in duplicate_columns)
        raise ValueError(
            "CSV has ambiguous columns after trimming whitespace: "
            f"{duplicate_text}"
        )
    out.columns = normalized_columns
    return _canonicalize_sequence_id_column(out)


def read_class_probability_csv(path: Path) -> pd.DataFrame:
    """Read classifier CSV output while preserving sequence ids as text."""

    return read_sequence_text_csv(path)


def _canonicalize_sequence_id_column(rows: pd.DataFrame) -> pd.DataFrame:
    """Add ``sequence_id`` when the input uses a supported sequence alias."""

    out = rows.copy()
    alias_keys = {alias.lower() for alias in SEQUENCE_ALIASES}
    source_columns = [
        column
        for column in out.columns
        if str(column).strip().lower() in alias_keys
    ]
    if len(source_columns) > 1:
        source_text = ", ".join(repr(str(column)) for column in source_columns)
        raise ValueError(
            "CSV has ambiguous sequence identifier columns: "
            f"{source_text}"
        )
    if not source_columns:
        return out

    source_column = source_columns[0]
    sequence_ids = _sequence_id_text(out[source_column])
    if source_column == "sequence_id":
        out["sequence_id"] = sequence_ids
    else:
        out.insert(0, "sequence_id", sequence_ids)
    return out


def _sequence_id_text(values: pd.Series) -> pd.Series:
    """Return stripped sequence ids without changing opaque text such as ``001``."""

    return values.where(values.notna(), "").astype(str).str.strip()
