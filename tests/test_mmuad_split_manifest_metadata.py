from __future__ import annotations

import json
from pathlib import Path

from raft_uav.mmuad.splits import load_split_manifest


def test_split_manifest_ignores_top_level_metadata_blocks(tmp_path: Path) -> None:
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "exported-splits-v1",
                "metadata": {"exported_by": "mmuad-tool", "source": "layout report"},
                "train": ["seq_train"],
                "val": ["seq_val"],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(manifest_path)

    assert manifest == {"train": ("seq_train",), "val": ("seq_val",)}


def test_split_manifest_ignores_metadata_inside_splits_mapping(tmp_path: Path) -> None:
    manifest_path = tmp_path / "splits.json"
    manifest_path.write_text(
        json.dumps(
            {
                "splits": {
                    "metadata": {"exported_by": "mmuad-tool", "source": "not a sequence"},
                    "train": ["seq_train"],
                    "val": {"sequences": [{"id": "seq_val"}]},
                }
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(manifest_path)

    assert manifest == {"train": ("seq_train",), "val": ("seq_val",)}
