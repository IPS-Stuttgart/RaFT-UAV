from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.io import JSON_TABLE_SUFFIXES, data_file_suffix, read_json_export_payload
from raft_uav.mmuad.radar_json_keys import (
    RADAR_HINT_KEYS,
    RADAR_NESTED_TABLE_KEYS,
    RADAR_PARENT_DEFAULT_KEYS,
    RADAR_SEQUENCE_KEYS,
    RADAR_TIME_KEYS,
)


def read_radar_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if data_file_suffix(path) in JSON_TABLE_SUFFIXES:
        return _records_to_frame(_json_radar_records(read_json_export_payload(path)), path=path)
    if data_file_suffix(path) == ".tsv":
        return pd.read_csv(path, sep="\t")
    if data_file_suffix(path) == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path)


def _json_radar_records(payload: Any) -> Any:
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in RADAR_NESTED_TABLE_KEYS:
        nested = _mapping_get_case_insensitive(payload, key)
        if nested is not None:
            return _records_from_nested_container(payload, nested)
    if _looks_like_column_map(payload) or _looks_like_row(payload):
        return payload
    return []


def _records_from_nested_container(parent: dict[Any, Any], nested: Any) -> Any:
    records = _json_radar_records(nested)
    defaults = _parent_defaults(parent)
    if not defaults:
        return records
    if isinstance(records, list):
        return [
            _merge_parent_defaults(defaults, record) if isinstance(record, dict) else record
            for record in records
        ]
    if isinstance(records, dict) and (_looks_like_column_map(records) or _looks_like_row(records)):
        return _merge_parent_defaults(defaults, records)
    return records


def _parent_defaults(parent: dict[Any, Any]) -> dict[str, Any]:
    return {
        key: value
        for key in RADAR_PARENT_DEFAULT_KEYS
        if (value := _mapping_get_case_insensitive(parent, key)) is not None
    }


def _merge_parent_defaults(defaults: dict[str, Any], record: dict[Any, Any]) -> dict[Any, Any]:
    row: dict[Any, Any] = {}
    record_has_time = _has_any_key(record, RADAR_TIME_KEYS)
    record_has_sequence = _has_any_key(record, RADAR_SEQUENCE_KEYS)
    for key, value in defaults.items():
        if key in RADAR_TIME_KEYS and record_has_time:
            continue
        if key in RADAR_SEQUENCE_KEYS and record_has_sequence:
            continue
        if not _has_any_key(record, (key,)):
            row[key] = value
    row.update(record)
    return row


def _records_to_frame(records: Any, *, path: Path | None = None) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records
    if isinstance(records, dict):
        if _looks_like_column_map(records):
            return pd.DataFrame(records)
        if _looks_like_row(records):
            return pd.DataFrame.from_records([records])
    if isinstance(records, list):
        if not records:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in records):
            return pd.DataFrame.from_records(records)
    label = str(path) if path is not None else "JSON payload"
    raise ValueError(f"radar polar JSON table {label} does not contain row objects")


def _mapping_get_case_insensitive(mapping: dict[Any, Any], key: str) -> Any | None:
    for candidate, value in mapping.items():
        if str(candidate).lower() == key.lower():
            return value
    return None


def _has_any_key(mapping: dict[Any, Any], keys: tuple[str, ...]) -> bool:
    present = {str(key).lower() for key in mapping}
    return any(key.lower() in present for key in keys)


def _looks_like_row(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    return bool(keys.intersection(RADAR_HINT_KEYS))


def _looks_like_column_map(payload: dict[Any, Any]) -> bool:
    keys = {str(key).lower() for key in payload}
    if not keys.intersection(RADAR_HINT_KEYS):
        return False
    return any(
        isinstance(value, (list, tuple))
        for key, value in payload.items()
        if str(key).lower() in RADAR_HINT_KEYS
    )
