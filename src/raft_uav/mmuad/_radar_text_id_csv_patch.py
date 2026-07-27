"""Preserve opaque radar CSV identifiers that resemble pandas NA tokens."""

from __future__ import annotations

from functools import wraps
from importlib import import_module
from pathlib import Path
from typing import Any, Collection

import pandas as pd

_PATCH_MARKER = "_raft_uav_radar_text_id_csv_patch_applied"


def _preserve_identifier_token(value: str) -> object:
    """Keep non-empty text verbatim while retaining blank-cell missingness."""

    return pd.NA if value == "" else value


def _read_csv_with_preserved_text_ids(
    path: Path,
    *,
    text_id_columns: Collection[str],
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a radar table without pandas interpreting text ids as NA sentinels."""

    header = pd.read_csv(path, nrows=0, **kwargs)
    normalized_id_columns = {str(column).strip().lower() for column in text_id_columns}
    identifier_columns = [
        column
        for column in header.columns
        if str(column).strip().lower() in normalized_id_columns
    ]
    if not identifier_columns:
        return pd.read_csv(path, **kwargs)

    read_kwargs = dict(kwargs)
    converters = dict(read_kwargs.pop("converters", {}) or {})
    for column in identifier_columns:
        converters.setdefault(column, _preserve_identifier_token)
    read_kwargs["converters"] = converters

    dtype = read_kwargs.get("dtype")
    if isinstance(dtype, dict):
        remaining_dtype = {
            column: value
            for column, value in dtype.items()
            if column not in identifier_columns
        }
        if remaining_dtype:
            read_kwargs["dtype"] = remaining_dtype
        else:
            read_kwargs.pop("dtype")

    rows = pd.read_csv(path, **read_kwargs)
    for column in identifier_columns:
        rows[column] = rows[column].astype("string")
    return rows


def install() -> None:
    """Patch the MMUAD radar CSV reader to preserve opaque text identifiers."""

    radar_module = import_module("raft_uav.mmuad.radar")
    if getattr(radar_module, _PATCH_MARKER, False):
        return

    original = radar_module._read_csv_preserving_text_ids

    @wraps(original)
    def _read_csv_preserving_text_ids(path: Path, **kwargs: Any) -> pd.DataFrame:
        return _read_csv_with_preserved_text_ids(
            path,
            text_id_columns=radar_module._RADAR_TEXT_ID_COLUMNS,
            **kwargs,
        )

    radar_module._read_csv_preserving_text_ids = _read_csv_preserving_text_ids
    setattr(radar_module, _PATCH_MARKER, True)
