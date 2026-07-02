from __future__ import annotations

import json
from pathlib import Path

from raft_uav.mmuad.splits import load_split_manifest, resolve_split_name, split_manifest_summary


def test_split_manifest_preserves_explicit_empty_mapping_splits(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "train": ["seq_a"],
                    "val": [],
                    "holdout": {"sequences": []},
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = load_split_manifest(path)

    assert manifest == {"train": ("seq_a",), "val": (), "holdout": ()}
    assert resolve_split_name(manifest, "VAL") == "val"
    assert split_manifest_summary(manifest)["holdout"] == {"count": 0, "sequence_ids": []}
