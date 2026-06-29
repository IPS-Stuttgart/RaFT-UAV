from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_reservoir_window_sweep.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_template_reservoir_window_sweep",
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
            "time_s": [9.96, 10.08, 20.2],
            "source": ["lidar_360", "livox_avia", "radar"],
            "track_id": ["raw", "translated", "late_radar"],
            "candidate_branch": ["raw", "source_translation", "radar"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [0.2, 0.9, 0.1],
            "confidence": [0.2, 0.9, 0.1],
        }
    )


def test_window_sweep_ranks_variants_by_template_coverage(tmp_path: Path) -> None:
    summary = window_sweep.run_template_reservoir_window_sweep(
        _candidate_rows(),
        _template_rows(),
        output_dir=tmp_path,
        max_time_delta_s_values=(0.1, 0.3),
        score_normalization_values=("none",),
        per_source_top_n=1,
        per_branch_top_n=1,
        global_top_n=1,
    )

    assert list(summary["max_time_delta_s"]) == [0.3, 0.1]
    assert int(summary.iloc[0]["missing_template_rows"]) == 0
    assert int(summary.iloc[1]["missing_template_rows"]) == 1
    for reservoir_csv in summary["reservoir_csv"]:
        assert Path(reservoir_csv).exists()


def test_window_sweep_cli_writes_summary_and_variants(tmp_path: Path) -> None:
    template = tmp_path / "template.csv"
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    _template_rows().to_csv(template, index=False)
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = window_sweep.main(
        [
            "--template-csv",
            str(template),
            "--candidate-csv",
            f"raw={raw}",
            "--candidate-csv",
            f"translated={translated}",
            "--output-dir",
            str(output),
            "--max-time-delta-s",
            "0.1,0.3",
            "--score-normalization",
            "none,window-rank",
            "--per-source-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
        ]
    )

    assert rc == 0
    summary_path = output / "mmuad_template_reservoir_window_sweep_summary.csv"
    json_path = output / "mmuad_template_reservoir_window_sweep_summary.json"
    assert summary_path.exists()
    assert json_path.exists()
    summary = pd.read_csv(summary_path)
    assert len(summary) == 4
    assert {"none", "window-rank"} <= set(summary["score_normalization"])
    assert int(summary["missing_template_rows"].min()) == 0
    for reservoir_csv in summary["reservoir_csv"]:
        assert Path(reservoir_csv).exists()
