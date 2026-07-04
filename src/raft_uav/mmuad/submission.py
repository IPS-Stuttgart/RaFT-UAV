"""Compatibility wrapper for MMUAD submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad import _submission_impl as _impl

_parse_original = _impl.parse_official_classification_cell
_load_sequence_class_map_original = _impl.load_sequence_class_map


def _parse_official_classification_cell_with_domain(value: Any) -> int:
    class_id = _parse_original(value)
    if class_id not in _impl.OFFICIAL_TRACK5_CLASS_IDS:
        raise ValueError(f"invalid official Track 5 class id: {class_id!r}")
    return class_id


_impl.parse_official_classification_cell = _parse_official_classification_cell_with_domain

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
parse_official_classification_cell = _parse_official_classification_cell_with_domain


def load_sequence_class_map(path: Path | None) -> dict[str, str]:
    """Load class maps while preserving textual CSV sequence identifiers."""

    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return _load_sequence_class_map_original(path)

    frame = pd.read_csv(path, dtype=str)
    lower = {str(col).lower(): col for col in frame.columns}
    rename = {}
    for alias in _impl._SEQUENCE_ID_ALIASES:
        if alias in lower:
            rename[lower[alias]] = "sequence_id"
            break
    for alias in _impl._UAV_TYPE_ALIASES:
        if alias in lower:
            rename[lower[alias]] = "uav_type"
            break
    frame = frame.rename(columns=rename)
    missing = {"sequence_id", "uav_type"}.difference(frame.columns)
    if missing:
        raise ValueError(f"class-map CSV missing columns: {sorted(missing)}")
    return {
        str(row["sequence_id"]): str(row["uav_type"])
        for _, row in frame.iterrows()
        if pd.notna(row["sequence_id"]) and pd.notna(row["uav_type"])
    }


_impl.load_sequence_class_map = load_sequence_class_map
