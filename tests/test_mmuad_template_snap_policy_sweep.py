from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_snap_policy_sweep.py"
spec = importlib.util.spec_from_file_location("mmuad_template_snap_policy_sweep", MODULE_PATH)
assert spec is not None and spec.loader is not None
policy_sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy_sweep
spec.loader.exec_module(policy_sweep)


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


def test_template_snap_policy_sweep_writes_one_bundle_per_policy(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _results().to_csv(results_csv, index=False)
    _template().to_csv(template_csv, index=False)

    summary = policy_sweep.run_template_snap_policy_sweep(
        results_path=results_csv,
        template_path=template_csv,
        output_dir=output_dir,
        resample_methods=("linear", "nearest"),
        max_interpolation_gap_s_values=(None,),
        classification_policies=("sequence-mode",),
    )

    assert len(summary) == 2
    assert summary["codabench_upload_ready"].all()
    for zip_path in summary["official_zip"]:
        assert Path(zip_path).exists()
    assert (output_dir / "mmuad_template_snap_policy_sweep_summary.csv").exists()
    assert (output_dir / "mmuad_template_snap_policy_sweep_summary.json").exists()


def test_template_snap_policy_sweep_cli_writes_summary(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _results().to_csv(results_csv, index=False)
    _template().to_csv(template_csv, index=False)

    rc = policy_sweep.main(
        [
            "--results",
            str(results_csv),
            "--template",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--resample-methods",
            "linear,nearest",
            "--max-interpolation-gap-s",
            "none",
            "--classification-policies",
            "sequence-mode",
            "--require-any-upload-ready",
        ]
    )

    assert rc == 0
    summary = pd.read_csv(output_dir / "mmuad_template_snap_policy_sweep_summary.csv")
    assert set(summary["resample_method"]) == {"linear", "nearest"}
    assert summary["codabench_upload_ready"].all()
    payload = json.loads((output_dir / "mmuad_template_snap_policy_sweep_summary.json").read_text())
    assert len(payload["rows"]) == 2
