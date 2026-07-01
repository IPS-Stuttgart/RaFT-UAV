from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_snap_official_results_to_template.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_snap_official_results_to_template",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
snapper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = snapper
spec.loader.exec_module(snapper)


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq001", "seq002"],
            "Timestamp": [0.0, 10.0, 30.0, 2.0],
            "Position": ["(0,0,0)", "(10,20,2)", "(30,60,6)", "(5,5,5)"],
            "Classification": [2, 2, 3, 1],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq001", "seq002"],
            "Timestamp": [0.0, 5.0, 10.0, 2.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [0, 0, 0, 0],
        }
    )


def test_snap_official_results_to_template_interpolates_and_keeps_sequence_class() -> None:
    snapped, diagnostics = snapper.snap_official_results_to_template(
        _results(),
        _template(),
        resample_method="linear",
        classification_policy="sequence-mode",
    )

    assert len(snapped) == 4
    midpoint = snapped.loc[
        (snapped["Sequence"] == "seq001") & (snapped["Timestamp"] == 5.0)
    ].iloc[0]
    assert midpoint["Position"] == "(5,10,1)"
    assert int(midpoint["Classification"]) == 2
    assert diagnostics["valid"].all()
    midpoint_method = diagnostics.loc[diagnostics["Timestamp"] == 5.0, "method"].iloc[0]
    assert midpoint_method == "linear"


def test_snap_official_results_to_template_rejects_fractional_classification_labels() -> None:
    results = _results()
    results.loc[0, "Classification"] = "1.5"

    with pytest.raises(ValueError, match="Classification"):
        snapper.snap_official_results_to_template(results, _template())


def test_snap_official_results_to_template_handles_empty_source_results() -> None:
    snapped, diagnostics = snapper.snap_official_results_to_template(
        pd.DataFrame(columns=["Sequence", "Timestamp", "Position", "Classification"]),
        pd.DataFrame({"Sequence": ["seq-missing"], "Timestamp": [1.0]}),
        missing_position_policy="zero",
    )

    assert snapped.to_dict("records") == [
        {
            "Sequence": "seq-missing",
            "Timestamp": 1.0,
            "Position": "(0,0,0)",
            "Classification": 0,
        }
    ]
    row = diagnostics.iloc[0]
    assert int(row["source_row_count"]) == 0
    assert row["method"] == "missing-zero"
    assert bool(row["valid"]) is False


def test_snap_official_results_to_template_nearest_classification_policy() -> None:
    snapped, _ = snapper.snap_official_results_to_template(
        _results(),
        pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [29.0]}),
        resample_method="nearest",
        classification_policy="nearest",
    )

    row = snapped.iloc[0]
    assert row["Position"] == "(30,60,6)"
    assert int(row["Classification"]) == 3


def test_snap_official_results_to_template_raises_on_missing_sequence() -> None:
    with pytest.raises(ValueError, match="no source results"):
        snapper.snap_official_results_to_template(
            _results(),
            pd.DataFrame({"Sequence": ["missing"], "Timestamp": [0.0]}),
            missing_position_policy="raise",
        )


def test_snap_official_results_cli_writes_upload_ready_zip(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _results().to_csv(results_csv, index=False)
    _template().to_csv(template_csv, index=False)

    rc = snapper.main(
        [
            "--results",
            str(results_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--require-leaderboard-ready",
        ]
    )

    assert rc == 0
    assert (output_dir / "mmaud_results.csv").exists()
    assert (output_dir / "ug2_submission.zip").exists()
    assert (output_dir / "mmuad_template_snap_diagnostics.csv").exists()
    validation_path = output_dir / "mmuad_template_snap_validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["leaderboard_ready"] is True
    assert validation["codabench_upload_ready"] is True
    manifest_path = output_dir / "mmuad_template_snap_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["row_count"] == 4
    assert manifest["source_result_rows"] == 4
    with ZipFile(output_dir / "ug2_submission.zip") as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
