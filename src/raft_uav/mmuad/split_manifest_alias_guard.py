"""Compatibility guard for whitespace-padded MMUAD split-manifest aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def install() -> None:
    """Patch split-manifest alias lookup to trim exported key/header whitespace."""

    from raft_uav.mmuad import splits as _splits

    def _normalized_key(value: Any) -> str:
        return str(value).strip().casefold()

    def _mapping_value_case_insensitive(mapping: Mapping[Any, Any], key: str) -> Any | None:
        lower_key = _normalized_key(key)
        for candidate, value in mapping.items():
            if _normalized_key(candidate) == lower_key:
                return value
        return None

    def _entry_value(entry: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
        normalized_keys = {_normalized_key(key): key for key in entry}
        for alias in aliases:
            key = alias if alias in entry else normalized_keys.get(_normalized_key(alias))
            if key is None:
                continue
            value = _splits._scalar_to_text(entry[key])
            if value is not None:
                return value
        return None

    def _split_values_to_sequence_ids(values: Any) -> tuple[str, ...]:
        out: list[str] = []
        if isinstance(values, dict):
            sequence_id = _entry_value(values, _splits._SEQUENCE_ID_ALIASES)
            if sequence_id is not None:
                return (sequence_id,)
            for key in _splits._SEQUENCE_LIST_KEYS:
                nested = _mapping_value_case_insensitive(values, key)
                if nested is not None:
                    return _split_values_to_sequence_ids(nested)
            for key, item in values.items():
                if _normalized_key(key) in _splits._SPLIT_VALUE_METADATA_KEYS:
                    continue
                if isinstance(item, dict):
                    sequence_id = _entry_value(item, _splits._SEQUENCE_ID_ALIASES)
                    if sequence_id is not None:
                        _splits._append_unique_value(out, sequence_id)
                        continue
                sequence_id = _splits._scalar_to_text(key)
                if sequence_id is not None:
                    _splits._append_unique_value(out, sequence_id)
            return tuple(out)
        if isinstance(values, list | tuple | set):
            for item in values:
                if isinstance(item, dict):
                    sequence_id = _entry_value(item, _splits._SEQUENCE_ID_ALIASES)
                    if sequence_id is not None:
                        _splits._append_unique_value(out, sequence_id)
                    continue
                sequence_id = _splits._scalar_to_text(item)
                if sequence_id is not None:
                    _splits._append_unique_value(out, sequence_id)
        return tuple(out)

    def _manifest_from_mapping(mapping: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for split, values in mapping.items():
            if _normalized_key(split) in _splits._SPLIT_VALUE_METADATA_KEYS:
                continue
            split_name = _splits._scalar_to_text(split)
            if split_name is None:
                continue
            ids = _split_values_to_sequence_ids(values)
            if ids or _splits._is_explicit_split_container(values):
                out[split_name] = list(ids)
        return {split: tuple(values) for split, values in out.items()}

    def _resolve_split_name(manifest: Mapping[str, Any], split_name: str) -> str:
        if split_name in manifest:
            return split_name
        normalized = _normalized_key(split_name)
        matches = [
            str(candidate)
            for candidate in manifest
            if _normalized_key(candidate) == normalized
        ]
        available = ", ".join(sorted(str(split) for split in manifest))
        if len(matches) == 1:
            return matches[0]
        if matches:
            joined = ", ".join(sorted(matches))
            raise ValueError(
                f"split {split_name!r} is ambiguous; case-insensitive matches: {joined}; "
                f"available splits: {available}"
            )
        raise ValueError(f"split {split_name!r} not found; available splits: {available}")

    _splits._mapping_value_case_insensitive = _mapping_value_case_insensitive
    _splits._entry_value = _entry_value
    _splits._split_values_to_sequence_ids = _split_values_to_sequence_ids
    _splits._manifest_from_mapping = _manifest_from_mapping
    _splits.resolve_split_name = _resolve_split_name
