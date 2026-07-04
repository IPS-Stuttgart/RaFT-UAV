"""Split-manifest helpers for exported MMUAD-style sequence roots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.sequence import SequencePaths


_SEQUENCE_ID_ALIASES = (
    "sequence_id",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "id",
    "name",
)
_SPLIT_ALIASES = ("split", "subset", "partition", "fold", "set")
_SEQUENCE_LIST_KEYS = ("sequence_ids", "sequences", "ids", "items", "sequence_names")
_SPLIT_VALUE_METADATA_KEYS = ("schema", "version", "description", "metadata", "meta")
_MISSING_TEXT_VALUES = frozenset({"", "nan", "none", "<na>", "nat"})


class _CaseInsensitiveSplitSummary(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if isinstance(key, str):
            resolved = resolve_split_name(self, key)
            return super().__getitem__(resolved)
        return super().__getitem__(key)


def load_split_manifest(path: Path) -> dict[str, tuple[str, ...]]:
    """Load a split manifest from JSON, YAML, or CSV.

    Supported JSON layouts::

        {"train": ["seq001"], "val": ["seq002"]}
        {"splits": {"train": ["seq001"], "val": ["seq002"]}}
        {"splits": {"train": {"sequences": [{"sequence_id": "seq001"}]}}}
        {"sequences": [{"sequence_id": "seq001", "split": "train"}]}

    Supported CSV layout::

        sequence_id,split
        seq001,train
        seq002,val

    CSV alias columns such as ``id,subset`` or ``name,partition`` are also
    accepted for exported MMUAD metadata files.
    """

    path = Path(path)
    if path.suffix.lower() in {".json", ".yaml", ".yml"}:
        payload = _load_manifest_payload(path)
        return _manifest_from_payload(payload)
    frame = pd.read_csv(path)
    manifest = _manifest_from_rows(frame.to_dict("records"))
    if not manifest:
        raise ValueError(
            "CSV split manifest must contain sequence id and split columns; "
            "accepted aliases include sequence_id/id/name and split/subset/partition"
        )
    return manifest


def _load_manifest_payload(path: Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except Exception:
        return json.loads(text)
    return yaml.safe_load(text)


def _mapping_value_case_insensitive(mapping: Mapping[Any, Any], key: str) -> Any | None:
    lower_key = key.lower()
    for candidate, value in mapping.items():
        if str(candidate).lower() == lower_key:
            return value
    return None


def resolve_split_name(manifest: Mapping[str, Any], split_name: str) -> str:
    """Return the manifest key matching ``split_name``.

    Exact keys are preferred, but case-only differences are accepted.  This keeps
    CLI options such as ``--split-name train`` usable with exported manifests
    whose split labels are capitalized as ``Train``/``Val``.
    """

    if split_name in manifest:
        return split_name
    lowered = str(split_name).lower()
    matches = [str(candidate) for candidate in manifest if str(candidate).lower() == lowered]
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


def _manifest_from_payload(payload: Any) -> dict[str, tuple[str, ...]]:
    if isinstance(payload, list):
        manifest = _manifest_from_rows(payload)
        if manifest:
            return manifest
        raise ValueError(
            "split manifest list entries must include sequence id and split fields"
        )
    if not isinstance(payload, dict):
        raise ValueError("split manifest must be an object or a list of sequence rows")

    single_row_manifest = _manifest_from_rows([payload])
    if single_row_manifest:
        return single_row_manifest

    splits = _mapping_value_case_insensitive(payload, "splits")
    if isinstance(splits, dict):
        manifest = _manifest_from_mapping(splits)
        if manifest:
            return manifest

    sequences = _mapping_value_case_insensitive(payload, "sequences")
    if isinstance(sequences, list):
        manifest = _manifest_from_rows(sequences)
        if manifest:
            return manifest
    if isinstance(sequences, dict):
        manifest = _manifest_from_mapping(sequences)
        if manifest:
            return manifest

    manifest = _manifest_from_mapping(payload)
    if manifest:
        return manifest
    raise ValueError("split manifest does not contain any split sequence ids")


def _manifest_from_mapping(mapping: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for split, values in mapping.items():
        if str(split).lower() in _SPLIT_VALUE_METADATA_KEYS:
            continue
        ids = _split_values_to_sequence_ids(values)
        if ids or _is_explicit_split_container(values):
            out[str(split)] = list(ids)
    return {split: tuple(values) for split, values in out.items()}


def _is_explicit_split_container(values: Any) -> bool:
    return isinstance(values, dict | list | tuple | set)


def _manifest_from_rows(rows: Iterable[Any]) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        split = _entry_value(row, _SPLIT_ALIASES)
        sequence_id = _entry_value(row, _SEQUENCE_ID_ALIASES)
        if split is None or sequence_id is None:
            continue
        _append_unique(out, split, sequence_id)
    return {split: tuple(values) for split, values in out.items()}


def _split_values_to_sequence_ids(values: Any) -> tuple[str, ...]:
    out: list[str] = []
    if isinstance(values, dict):
        sequence_id = _entry_value(values, _SEQUENCE_ID_ALIASES)
        if sequence_id is not None:
            return (sequence_id,)
        for key in _SEQUENCE_LIST_KEYS:
            nested = _mapping_value_case_insensitive(values, key)
            if nested is not None:
                return _split_values_to_sequence_ids(nested)
        for key, item in values.items():
            if str(key).lower() in _SPLIT_VALUE_METADATA_KEYS:
                continue
            if isinstance(item, dict):
                sequence_id = _entry_value(item, _SEQUENCE_ID_ALIASES)
                if sequence_id is not None:
                    _append_unique_value(out, sequence_id)
                    continue
            sequence_id = _scalar_to_text(key)
            if sequence_id is not None:
                _append_unique_value(out, sequence_id)
        return tuple(out)
    if isinstance(values, list | tuple | set):
        for item in values:
            if isinstance(item, dict):
                sequence_id = _entry_value(item, _SEQUENCE_ID_ALIASES)
                if sequence_id is not None:
                    _append_unique_value(out, sequence_id)
                continue
            sequence_id = _scalar_to_text(item)
            if sequence_id is not None:
                _append_unique_value(out, sequence_id)
    return tuple(out)


def _entry_value(entry: Mapping[str, Any], aliases: tuple[str, ...]) -> str | None:
    lower_keys = {str(key).lower(): key for key in entry}
    for alias in aliases:
        key = alias if alias in entry else lower_keys.get(alias)
        if key is None:
            continue
        value = _scalar_to_text(entry[key])
        if value is not None:
            return value
    return None


def _scalar_to_text(value: Any) -> str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool):
        return None
    try:
        missing = pd.isna(value)
    except TypeError:
        missing = False
    if isinstance(missing, bool) and missing:
        return None
    if not isinstance(value, str | int | float):
        return None
    text = str(value).strip()
    if text.lower() in _MISSING_TEXT_VALUES:
        return None
    return text


def _append_unique(mapping: dict[str, list[str]], key: str, value: str) -> None:
    bucket = mapping.setdefault(key, [])
    _append_unique_value(bucket, value)


def _append_unique_value(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def filter_sequences_by_split(
    sequences: list[SequencePaths],
    manifest: dict[str, tuple[str, ...]],
    split_name: str,
) -> list[SequencePaths]:
    """Return only sequences listed in ``split_name`` of ``manifest``."""

    resolved_split_name = resolve_split_name(manifest, split_name)
    wanted = manifest[resolved_split_name]
    return [
        sequence
        for sequence in sequences
        if any(_sequence_matches_manifest_reference(sequence, reference) for reference in wanted)
    ]


def _sequence_matches_manifest_reference(sequence: SequencePaths, reference: str) -> bool:
    normalized = _normalize_sequence_reference(reference)
    if not normalized:
        return False
    sequence_id = _normalize_sequence_reference(sequence.sequence_id)
    root_name = _normalize_sequence_reference(sequence.root.name)
    if normalized in {sequence_id, root_name}:
        return True
    root_path = _normalize_sequence_reference(sequence.root.as_posix())
    return root_path == normalized or root_path.endswith(f"/{normalized}")


def _normalize_sequence_reference(value: str) -> str:
    text = str(value).strip().replace("\\", "/").strip("/")
    while text.startswith("./"):
        text = text[2:]
    while "//" in text:
        text = text.replace("//", "/")
    return text


def filter_sequences_by_split_folder(
    sequences: list[SequencePaths],
    root: Path,
    split_name: str,
) -> list[SequencePaths]:
    """Return sequences whose path is under a top-level split folder.

    This supports MMUAD-style roots such as ``train/seq001`` and ``val/seq002``
    when no explicit split manifest has been exported.
    """

    root = Path(root)
    wanted = str(split_name)
    wanted_casefold = wanted.casefold()
    out: list[SequencePaths] = []
    for sequence in sequences:
        try:
            parts = sequence.root.relative_to(root).parts
        except ValueError:
            parts = sequence.root.parts
        if parts and str(parts[0]).casefold() == wanted_casefold:
            out.append(sequence)
        elif sequence.root.name.casefold() == wanted_casefold:
            out.append(sequence)
    return out


def split_manifest_summary(manifest: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Return count summary for provenance files."""

    return _CaseInsensitiveSplitSummary(
        {
            split: {"count": len(values), "sequence_ids": list(values)}
            for split, values in manifest.items()
        }
    )
