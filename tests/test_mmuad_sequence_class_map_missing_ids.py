from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import mmuad_sequence_class_map_from_predictions as class_map_tool  # noqa: E402


def test_sequence_class_map_drops_missing_like_sequence_ids() -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq001", None, "None", "<NA>", pd.NA, ""],
            "classification": [2, 1, 1, 1, 1, 1],
        }
    )

    class_map, diagnostics = class_map_tool.build_sequence_class_map_from_predictions(predictions)

    assert class_map.to_dict("records") == [{"sequence_id": "seq001", "uav_type": 2}]
    assert diagnostics["sequence_id"].tolist() == ["seq001"]
