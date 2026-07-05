from __future__ import annotations

import importlib
from pathlib import Path

from raft_uav.mmuad import submission


def test_submission_wrapper_reload_is_idempotent(tmp_path: Path) -> None:
    reloaded = importlib.reload(submission)
    reloaded = importlib.reload(reloaded)

    assert reloaded.parse_official_classification_cell("2") == 2

    class_map_path = tmp_path / "class_map.csv"
    class_map_path.write_text("sequence_id,uav_type\nseq0001,2\n", encoding="utf-8")

    assert reloaded.load_sequence_class_map(class_map_path) == {"seq0001": "2"}
