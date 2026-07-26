from __future__ import annotations

import json
from pathlib import Path

from raft_uav.mmuad.layout import inspect_mmuad_layout


def test_layout_inspector_handles_null_topic_map_exports(tmp_path: Path) -> None:
    (tmp_path / "topic_map.json").write_text(
        json.dumps(
            {
                "schema": "raft-uav-mmuad-topic-map-v1",
                "exports": None,
            }
        ),
        encoding="utf-8",
    )

    summary = inspect_mmuad_layout(tmp_path)

    assert summary["file_count"] == 1
    assert summary["category_counts"] == {"json_metadata": 1}
    assert summary["sequence_candidates"] == [
        {
            "sequence_id": ".",
            "file_count": 1,
            "categories": {"json_metadata": 1},
            "has_topic_map_export": False,
            "has_native_topic_map": False,
            "has_candidates_or_points": False,
            "has_truth_or_labels": False,
            "has_class_labels": False,
            "has_calibration": False,
        }
    ]
