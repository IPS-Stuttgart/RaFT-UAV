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
    assert summary["sequence_candidates"][0]["has_topic_map_export"] is False
    assert summary["sequence_candidates"][0]["has_truth_or_labels"] is False
