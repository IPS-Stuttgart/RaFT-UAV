"""Compatibility wrapper for MMUAD submission helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from raft_uav.mmuad import _submission_impl as _impl

_ORIGINAL_PARSE_ATTR = "_raft_uav_original_parse_official_classification_cell"
_ORIGINAL_LOAD_ATTR = "_raft_uav_original_load_sequence_class_map"

if not hasattr(_impl, _ORIGINAL_PARSE_ATTR):
    setattr(_impl, _ORIGINAL_PARSE_ATTR, _impl.parse_official_classification_cell)
if not hasattr(_impl, _ORIGINAL_LOAD_ATTR):
    setattr(_impl, _ORIGINAL_LOAD_ATTR, _impl.load_sequence_class_map)

_parse_original = getattr(_impl, _ORIGINAL_PARSE_ATTR)
_load_sequence_class_map_original = getattr(_impl, _ORIGINAL_LOAD_ATTR)


def _parse_official_classification_cell_with_domain(value: Any) -> int:
    class_id = _parse_original(value)
    if class_id not in _impl.OFFICIAL_TRACK5_CLASS_IDS:
        allowed = ", ".join(str(item) for item in sorted(_impl.OFFICIAL_TRACK5_CLASS_IDS))
        raise ValueError(
            "official Track 5 classification must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return class_id


def _load_sequence_class_map_with_official_sequences(path: Path | str | None) -> dict[str, str]:
    """Load class maps while canonicalizing CSV/JSON/YAML sequence ids like official rows."""

    if path is None:
        return {}
    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        return _canonicalize_sequence_class_map(_load_sequence_class_map_original(path))

    frame = _impl.pd.read_csv(path, dtype=str, keep_default_na=False)
    lower = {str(col).lower(): col for col in frame.columns}
    rename: dict[Any, str] = {}
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

    class_map: dict[str, str] = {}
    for _, row in frame.iterrows():
        sequence_id = _class_map_sequence_key(row["sequence_id"])
        uav_type = _class_map_uav_type(row["uav_type"])
        if sequence_id is not None and uav_type is not None:
            class_map[sequence_id] = uav_type
    return class_map


def _canonicalize_sequence_class_map(class_map: dict[Any, Any]) -> dict[str, str]:
    """Drop missing-like sequence IDs from non-CSV class-map formats."""

    normalized: dict[str, str] = {}
    for sequence_id, uav_type in class_map.items():
        normalized_sequence_id = _class_map_sequence_key(sequence_id)
        normalized_uav_type = _class_map_uav_type(uav_type)
        if normalized_sequence_id is not None and normalized_uav_type is not None:
            normalized[normalized_sequence_id] = normalized_uav_type
    return normalized


def _class_map_sequence_key(value: Any) -> str | None:
    try:
        return _impl.parse_official_sequence_cell(value)
    except ValueError:
        return None


def _class_map_uav_type(value: Any) -> str | None:
    if isinstance(value, _impl.np.generic):
        value = value.item()
    return _impl._scalar_to_text(value)


_impl.parse_official_classification_cell = _parse_official_classification_cell_with_domain
_impl.load_sequence_class_map = _load_sequence_class_map_with_official_sequences

globals().update(
    {
        name: getattr(_impl, name)
        for name in dir(_impl)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
parse_official_classification_cell = _parse_official_classification_cell_with_domain
load_sequence_class_map = _load_sequence_class_map_with_official_sequences
