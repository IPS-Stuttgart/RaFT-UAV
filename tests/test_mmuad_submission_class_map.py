from __future__ import annotations

import json

from raft_uav.mmuad.submission import load_sequence_class_map


def test_json_sequence_class_map_filters_missing_like_sequence_ids(tmp_path) -> None:
    path = tmp_path / "class_map.json"
    path.write_text(
        json.dumps(
            {
                "seq-valid": 2,
                "none": 1,
                "nan": 3,
                "<NA>": 0,
            }
        ),
        encoding="utf-8",
    )

    assert load_sequence_class_map(path) == {"seq-valid": "2"}
