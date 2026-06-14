from __future__ import annotations

import json
from pathlib import Path

from raft_uav.mmuad.splits import load_split_manifest


def test_split_manifest_accepts_case_insensitive_split_container_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "Splits": {
                    "Train": {
                        "Sequences": [
                            {"Sequence_ID": "seq_train_a"},
                            {"ID": "seq_train_b"},
                        ]
                    },
                    "Val": {"Sequence_IDs": ["seq_val_a", "seq_val_b"]},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(path)

    assert manifest == {
        "Train": ("seq_train_a", "seq_train_b"),
        "Val": ("seq_val_a", "seq_val_b"),
    }


def test_split_manifest_accepts_case_insensitive_top_level_sequence_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "Sequences": [
                    {"Sequence": "seq_train", "Subset": "train"},
                    {"ID": "seq_test", "Fold": "test"},
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(path)

    assert manifest == {"train": ("seq_train",), "test": ("seq_test",)}
