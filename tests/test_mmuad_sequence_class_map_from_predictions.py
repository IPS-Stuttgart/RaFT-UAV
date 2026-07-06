from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_sequence_class_map_from_predictions.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_sequence_class_map_from_predictions",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
class_map_tool = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = class_map_tool
spec.loader.exec_module(class_map_tool)


def _generic_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001", "seq002", "seq002"],
            "time_s": [0.0, 1.0, 2.0, 0.0, 1.0],
            "classification": [2, 2, 3, 1, 1],
            "classification_confidence": [0.3, 0.4, 0.99, 0.6, 0.7],
        }
    )


def _official_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": ["(0,0,0)", "(1,1,1)", "(2,2,2)"],
            "Classification": [3, 3, 0],
        }
    )


def test_build_sequence_class_map_uses_mode_policy_by_default() -> None:
    class_map, diagnostics = class_map_tool.build_sequence_class_map_from_predictions(
        _generic_predictions(),
    )

    assert class_map.set_index("sequence_id").loc["seq001", "uav_type"] == 2
    assert class_map.set_index("sequence_id").loc["seq002", "uav_type"] == 1
    seq001 = diagnostics.set_index("sequence_id").loc["seq001"]
    assert seq001["selected_class_count"] == 2
    assert seq001["class_count_3"] == 1


def test_build_sequence_class_map_can_select_by_confidence() -> None:
    class_map, diagnostics = class_map_tool.build_sequence_class_map_from_predictions(
        _generic_predictions(),
        policy="confidence",
    )

    assert class_map.set_index("sequence_id").loc["seq001", "uav_type"] == 3
    seq001 = diagnostics.set_index("sequence_id").loc["seq001"]
    assert seq001["policy"] == "confidence"
    assert seq001["selected_class_count"] == 1


def test_build_sequence_class_map_rejects_non_track5_class_id() -> None:
    bad = pd.DataFrame({"sequence_id": ["seq001"], "classification": [9]})

    with pytest.raises(ValueError, match="classification must be one of"):
        class_map_tool.build_sequence_class_map_from_predictions(bad)


def test_load_prediction_class_rows_reads_official_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "submission.zip"
    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("mmaud_results.csv", _official_results().to_csv(index=False))

    rows = class_map_tool.load_prediction_class_rows(zip_path)
    class_map, _ = class_map_tool.build_sequence_class_map_from_predictions(rows)

    assert class_map.set_index("sequence_id").loc["seq001", "uav_type"] == 3
    assert class_map.set_index("sequence_id").loc["seq002", "uav_type"] == 0


def test_load_prediction_class_rows_preserves_zero_padded_generic_sequences(
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "sequence_id,time_s,classification\n"
        "001,0.0,2\n"
        "001,1.0,2\n"
        "010,0.0,3\n",
        encoding="utf-8",
    )

    rows = class_map_tool.load_prediction_class_rows(predictions)
    class_map, _ = class_map_tool.build_sequence_class_map_from_predictions(rows)

    assert rows["sequence_id"].tolist() == ["001", "001", "010"]
    assert class_map.set_index("sequence_id").loc["001", "uav_type"] == 2
    assert class_map.set_index("sequence_id").loc["010", "uav_type"] == 3


def test_sequence_class_map_cli_writes_artifacts(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    output_csv = tmp_path / "class_map.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    summary_json = tmp_path / "summary.json"
    _generic_predictions().to_csv(predictions, index=False)

    rc = class_map_tool.main(
        [
            "--predictions",
            str(predictions),
            "--output-csv",
            str(output_csv),
            "--diagnostics-csv",
            str(diagnostics_csv),
            "--summary-json",
            str(summary_json),
        ]
    )

    assert rc == 0
    assert output_csv.exists()
    assert diagnostics_csv.exists()
    payload = json.loads(summary_json.read_text())
    assert payload["sequence_count"] == 2
    assert payload["class_histogram"] == {"1": 1, "2": 1}
    class_map = pd.read_csv(output_csv)
    assert list(class_map.columns) == ["sequence_id", "uav_type"]
