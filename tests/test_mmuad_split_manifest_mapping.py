from __future__ import annotations

import json
from pathlib import Path

from raft_uav.mmuad.splits import load_split_manifest


def test_split_manifest_accepts_mapping_keyed_by_sequence_id(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    path.write_text(
        json.dumps(
            {
                "splits": {
                    "dev": {
                        "seq_a": {"frames": 10},
                        "seq_b": {"frames": 11},
                    },
                    "eval": {
                        "sequences": {
                            "seq_c": {"frames": 5},
                            "seq_d": {},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert load_split_manifest(path) == {
        "dev": ("seq_a", "seq_b"),
        "eval": ("seq_c", "seq_d"),
    }
