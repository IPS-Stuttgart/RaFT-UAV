from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_snap_grid.py"
spec = importlib.util.spec_from_file_location("mmuad_template_snap_grid", MODULE_PATH)
assert spec is not None and spec.loader is not None
snap_grid = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = snap_grid
spec.loader.exec_module(snap_grid)


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001", "seq002"],
            "Timestamp": [0.0, 10.0, 2.0],
            "Position": ["(0,0,0)", "(10,20,2)", "(5,5,5)"],
            "Classification": [2, 2, 1],
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


def test_template_snap_grid_writes_ranked_variants(tmp_path: Path) -> None:
    summary = snap_grid.run_template_snap_grid(
        results=_results(),
        template=_template(),
        output_dir=tmp_path,
        resample_methods=("linear", "nearest"),
        max_interpolation_gaps_s=(None,),
        classification_policies=("sequence-mode",),
    )

    assert len(summary) == 2
    assert set(summary["resample_method"]) == {"linear", "nearest"}
    assert summary["leaderboard_ready"].all()
    assert summary["codabench_upload_ready"].all()
    for path in summary["official_zip"]:
        assert Path(path).exists()
    assert (tmp_path / "mmuad_template_snap_grid_summary.csv").exists()
    payload = json.loads((tmp_path / "mmuad_template_snap_grid_summary.json").read_text())
    assert len(payload["rows"]) == 2


def test_template_snap_grid_cli_writes_summary(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _results().to_csv(results_csv, index=False)
    _template().to_csv(template_csv, index=False)

    rc = snap_grid.main(
        [
            "--results",
            str(results_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--resample-method",
            "linear",
            "--resample-method",
            "nearest",
            "--max-interpolation-gap-s",
            "none,4.0",
            "--classification-policy",
            "sequence-mode",
            "--require-at-least-one-leaderboard-ready",
        ]
    )

    assert rc == 0
    summary = pd.read_csv(output_dir / "mmuad_template_snap_grid_summary.csv")
    assert len(summary) == 4
    assert summary["leaderboard_ready"].any()
    assert "pose_mse_loss_m2" in summary.columns
    assert (output_dir / "mmuad_template_snap_grid_summary.json").exists()
