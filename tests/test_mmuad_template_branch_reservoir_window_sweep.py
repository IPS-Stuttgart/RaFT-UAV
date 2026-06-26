from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_branch_reservoir_window_sweep.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_template_branch_reservoir_window_sweep",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
window_sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = window_sweep
spec.loader.exec_module(window_sweep)


def _template_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001", "seq001"],
            "Timestamp": [10.0, 20.0],
            "Position": ["(0,0,0)", "(0,0,0)"],
            "Classification": [0, 0],
        }
    )


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [9.96, 10.16, 20.4],
            "source": ["lidar_360", "livox_avia", "radar"],
            "track_id": ["near", "medium", "late"],
            "candidate_branch": ["raw", "source_translation", "radar"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [0.5, 0.9, 0.7],
            "confidence": [0.5, 0.9, 0.7],
        }
    )


def test_template_window_sweep_reports_coverage_tradeoff(tmp_path: Path) -> None:
    summary = window_sweep.run_template_window_sweep(
        _candidate_rows(),
        _template_rows(),
        output_dir=tmp_path,
        max_time_delta_values_s=(0.05, 0.2),
        per_source_top_n=1,
        per_branch_top_n=1,
        global_top_n=1,
    )

    small = summary.loc[summary["max_time_delta_s"] == 0.05].iloc[0]
    large = summary.loc[summary["max_time_delta_s"] == 0.2].iloc[0]
    assert int(small["covered_template_rows"]) == 1
    assert int(large["covered_template_rows"]) == 1
    assert float(large["mean_reservoir_count"]) > float(small["mean_reservoir_count"])
    assert Path(large["reservoir_csv"]).exists()


def test_template_window_sweep_cli_writes_summary(tmp_path: Path) -> None:
    template = tmp_path / "template.csv"
    raw = tmp_path / "raw.csv"
    output = tmp_path / "out"
    _template_rows().to_csv(template, index=False)
    _candidate_rows().to_csv(raw, index=False)

    rc = window_sweep.main(
        [
            "--template-csv",
            str(template),
            "--candidate-csv",
            f"raw={raw}",
            "--output-dir",
            str(output),
            "--max-time-delta-s",
            "0.05,0.2",
            "--per-source-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
        ]
    )

    assert rc == 0
    summary_csv = output / "mmuad_template_branch_reservoir_window_sweep_summary.csv"
    summary_json = output / "mmuad_template_branch_reservoir_window_sweep_summary.json"
    assert summary_csv.exists()
    assert summary_json.exists()
    summary = pd.read_csv(summary_csv)
    assert set(summary["window_label"]) == {"window_0p05s", "window_0p2s"}
    payload = json.loads(summary_json.read_text())
    assert len(payload["rows"]) == 2
