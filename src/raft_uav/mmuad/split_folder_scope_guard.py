"""Compatibility guard for split-folder filtering outside a sequence root."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def patch_module(split_module: Any) -> None:
    """Keep split-folder selection confined to the requested sequence root."""

    def filter_sequences_by_split_folder(
        sequences: list[Any],
        root: Path,
        split_name: str,
    ) -> list[Any]:
        root_path = Path(root).resolve()
        wanted_casefold = str(split_name).casefold()
        selected: list[Any] = []
        for sequence in sequences:
            sequence_root = Path(sequence.root).resolve()
            try:
                relative = sequence_root.relative_to(root_path)
            except ValueError:
                continue
            parts = relative.parts
            if parts and str(parts[0]).casefold() == wanted_casefold:
                selected.append(sequence)
            elif not parts and root_path.name.casefold() == wanted_casefold:
                selected.append(sequence)
        return selected

    split_module.filter_sequences_by_split_folder = filter_sequences_by_split_folder
