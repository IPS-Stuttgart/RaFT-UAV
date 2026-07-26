from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "mmuad_sequence_class_map_from_predictions.py"
)
spec = importlib.util.spec_from_file_location(
    "mmuad_sequence_class_map_last_policy",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
class_map_tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = class_map_tool
spec.loader.exec_module(class_map_tool)


def _selected_class(predictions: pd.DataFrame) -> int:
    class_map, _ = class_map_tool.build_sequence_class_map_from_predictions(
        predictions,
        policy="last",
    )
    return int(class_map.loc[0, "uav_type"])


def test_last_policy_is_order_invariant_for_equal_latest_timestamps() -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [9.0, 10.0, 10.0],
            "classification": [3, 2, 1],
            "classification_confidence": [0.99, 0.2, 0.8],
        }
    )

    assert _selected_class(predictions) == 1
    assert _selected_class(predictions.iloc[::-1]) == 1


def test_last_policy_uses_class_id_as_final_tie_break() -> None:
    predictions = pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [10.0, 10.0],
            "classification": [2, 1],
            "classification_confidence": [0.8, 0.8],
        }
    )

    assert _selected_class(predictions) == 1
    assert _selected_class(predictions.iloc[::-1]) == 1
