from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "mmuad_sequence_class_map_from_predictions.py"
)
spec = importlib.util.spec_from_file_location(
    "mmuad_sequence_class_map_last_policy_inputs",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
class_map_tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = class_map_tool
spec.loader.exec_module(class_map_tool)


def _predictions(timestamps: list[object] | None) -> pd.DataFrame:
    data: dict[str, list[object]] = {
        "sequence_id": ["seq001", "seq001"],
        "classification": [1, 2],
        "classification_confidence": [0.9, 0.1],
    }
    if timestamps is not None:
        data["time_s"] = timestamps
    return pd.DataFrame(data)


def test_last_policy_requires_timestamp_column() -> None:
    with pytest.raises(ValueError, match="requires a timestamp column"):
        class_map_tool.build_sequence_class_map_from_predictions(
            _predictions(None),
            policy="last",
        )


@pytest.mark.parametrize(
    "timestamps",
    [
        [0.0, "not-a-time"],
        [0.0, np.nan],
        [0.0, np.inf],
        [0.0, True],
        [0.0, [1.0]],
    ],
)
def test_last_policy_rejects_invalid_timestamps(timestamps: list[object]) -> None:
    with pytest.raises(ValueError, match="finite real timestamps"):
        class_map_tool.build_sequence_class_map_from_predictions(
            _predictions(timestamps),
            policy="last",
        )


def test_last_policy_accepts_numeric_timestamp_text() -> None:
    class_map, _ = class_map_tool.build_sequence_class_map_from_predictions(
        _predictions(["1.0", "2.0"]),
        policy="last",
    )

    assert class_map.to_dict("records") == [{"sequence_id": "seq001", "uav_type": 2}]


def test_other_policies_keep_optional_timestamp_behavior() -> None:
    class_map, _ = class_map_tool.build_sequence_class_map_from_predictions(
        _predictions(None),
        policy="confidence",
    )

    assert class_map.to_dict("records") == [{"sequence_id": "seq001", "uav_type": 1}]
