"""Split-manifest helpers for exported MMUAD-style sequence roots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.sequence import SequencePaths


def load_split_manifest(path: Path) -> dict[str, tuple[str, ...]]:
    """Load a simple split manifest from JSON or CSV.

    Supported JSON layouts::

        {"train": ["seq001"], "val": ["seq002"]}
        {"splits": {"train": ["seq001"], "val": ["seq002"]}}

    Supported CSV layout::

        sequence_id,split
        seq001,train
        seq002,val
    """

    path = Path(path)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "splits" in payload and isinstance(payload["splits"], dict):
            payload = payload["splits"]
        return {
            str(split): tuple(str(item) for item in values)
            for split, values in payload.items()
            if isinstance(values, list)
        }
    frame = pd.read_csv(path)
    if "sequence_id" not in frame.columns or "split" not in frame.columns:
        raise ValueError("CSV split manifest must contain sequence_id and split columns")
    out: dict[str, list[str]] = {}
    for _, row in frame.iterrows():
        out.setdefault(str(row["split"]), []).append(str(row["sequence_id"]))
    return {split: tuple(values) for split, values in out.items()}


def filter_sequences_by_split(
    sequences: list[SequencePaths],
    manifest: dict[str, tuple[str, ...]],
    split_name: str,
) -> list[SequencePaths]:
    """Return only sequences listed in ``split_name`` of ``manifest``."""

    if split_name not in manifest:
        available = ", ".join(sorted(manifest))
        raise ValueError(f"split {split_name!r} not found; available splits: {available}")
    wanted = set(manifest[split_name])
    return [sequence for sequence in sequences if sequence.sequence_id in wanted]


def split_manifest_summary(manifest: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """Return count summary for provenance files."""

    return {split: {"count": len(values), "sequence_ids": list(values)} for split, values in manifest.items()}
