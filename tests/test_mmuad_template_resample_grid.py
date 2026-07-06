from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_resample_grid.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_template_resample_grid",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
resample_grid = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = resample_grid
spec.loader.exec_module(resample_grid)


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq002"],
            "time_s": [0.0, 10.0, 0.0],
            "state_x_m": [0.0, 10.0, 5.0],
            "state_y_m": [0.0, 20.0, 5.0],
            "state_z_m": [1.0, 3.0, 7.0],
            "classification": [2, 2, 1],
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq001", "seq002"],
            "Timestamp": [0.0, 5.0, 10.0, 0.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [0, 0, 0, 0],
        }
    )


def test_template_resample_grid_writes_variants_and_preserves_classification(
    tmp_path: Path,
) -> None:
    summary = resample_grid.run_template_resample_grid(
        estimates=_estimates(),
        template=_template(),
        output_dir=tmp_path,
        resample_methods=("linear", "nearest"),
        max_interpolation_gaps_s=(None,),
        classification_policies=("sequence-mode",),
    )

    assert len(summary) == 2
    assert (tmp_path / "mmuad_template_resample_grid_summary.csv").exists()
    assert (tmp_path / "mmuad_template_resample_grid_summary.json").exists()
    assert set(summary["codabench_upload_ready"]) == {True}
    linear = summary.loc[summary["resample_method"] == "linear"].iloc[0]
    official_rows = pd.read_csv(linear["official_results_csv"])
    assert official_rows["Classification"].tolist() == [2, 2, 2, 1]
    with ZipFile(linear["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_template_resample_grid_cli_writes_summary(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _estimates().to_csv(estimates_csv, index=False)
    _template().to_csv(template_csv, index=False)

    rc = resample_grid.main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--resample-method",
            "linear,nearest",
            "--max-interpolation-gap-s",
            "none,4",
            "--classification-policy",
            "sequence-mode",
            "--require-leaderboard-ready",
        ]
    )

    assert rc == 0
    summary_path = output_dir / "mmuad_template_resample_grid_summary.csv"
    summary = pd.read_csv(summary_path)
    assert len(summary) == 4
    assert set(summary["classification_policy"]) == {"sequence-mode"}
    assert summary["codabench_upload_ready"].all()
    summary_json = output_dir / "mmuad_template_resample_grid_summary.json"
    payload = json.loads(summary_json.read_text())
    assert len(payload["rows"]) == 4
