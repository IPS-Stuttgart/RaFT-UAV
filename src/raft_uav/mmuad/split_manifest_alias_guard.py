"""Compatibility guard for whitespace-padded MMUAD split-manifest aliases."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _normalized_key(value: Any) -> str:
    return str(value).strip().casefold()


def patch_module(split_module: Any) -> None:
    """Patch a split-manifest module to trim exported key/header whitespace."""

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
            value = split_module._scalar_to_text(entry[key])
            if value is not None:
                return value
        return None

    def _split_values_to_sequence_ids(values: Any) -> tuple[str, ...]:
        out: list[str] = []
        if isinstance(values, dict):
            sequence_id = _entry_value(values, split_module._SEQUENCE_ID_ALIASES)
            if sequence_id is not None:
                return (sequence_id,)
            for key in split_module._SEQUENCE_LIST_KEYS:
                nested = _mapping_value_case_insensitive(values, key)
                if nested is not None:
                    return _split_values_to_sequence_ids(nested)
            for key, item in values.items():
                if _normalized_key(key) in split_module._SPLIT_VALUE_METADATA_KEYS:
                    continue
                if isinstance(item, dict):
                    sequence_id = _entry_value(item, split_module._SEQUENCE_ID_ALIASES)
                    if sequence_id is not None:
                        split_module._append_unique_value(out, sequence_id)
                        continue
                sequence_id = split_module._scalar_to_text(key)
                if sequence_id is not None:
                    split_module._append_unique_value(out, sequence_id)
            return tuple(out)
        if isinstance(values, list | tuple | set):
            for item in values:
                if isinstance(item, dict):
                    sequence_id = _entry_value(item, split_module._SEQUENCE_ID_ALIASES)
                    if sequence_id is not None:
                        split_module._append_unique_value(out, sequence_id)
                    continue
                sequence_id = split_module._scalar_to_text(item)
                if sequence_id is not None:
                    split_module._append_unique_value(out, sequence_id)
        return tuple(out)

    def _manifest_from_mapping(mapping: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        out: dict[str, list[str]] = {}
        for split, values in mapping.items():
            if _normalized_key(split) in split_module._SPLIT_VALUE_METADATA_KEYS:
                continue
            split_name = split_module._scalar_to_text(split)
            if split_name is None:
                continue
            ids = _split_values_to_sequence_ids(values)
            if ids or split_module._is_explicit_split_container(values):
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

    split_module._mapping_value_case_insensitive = _mapping_value_case_insensitive
    split_module._entry_value = _entry_value
    split_module._split_values_to_sequence_ids = _split_values_to_sequence_ids
    split_module._manifest_from_mapping = _manifest_from_mapping
    split_module.resolve_split_name = _resolve_split_name


def install() -> None:
    from raft_uav.mmuad import splits as _splits

    patch_module(_splits)
