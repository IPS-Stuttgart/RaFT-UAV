from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_branch_reservoir_timebin_sweep.py"
spec = importlib.util.spec_from_file_location("mmuad_branch_reservoir_timebin_sweep", MODULE_PATH)
assert spec is not None and spec.loader is not None
timebin_sweep = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = timebin_sweep
spec.loader.exec_module(timebin_sweep)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001"],
            "time_s": [0.0, 0.04, 0.08],
            "source": ["lidar_360", "livox_avia", "radar"],
            "track_id": ["raw", "translated", "radar"],
            "candidate_branch": ["raw", "source_translation", "radar"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "ranker_score": [0.9, 0.5, 0.4],
            "confidence": [0.9, 0.5, 0.4],
        }
    )


def test_timebin_sweep_preserves_original_candidate_timestamps(tmp_path: Path) -> None:
    summary = timebin_sweep.run_timebin_sweep(
        _candidate_rows(),
        output_dir=tmp_path,
        time_bins_s=(0.1,),
        per_source_top_n=1,
        per_branch_top_n=1,
        global_top_n=1,
    )

    reservoir_csv = Path(summary.loc[0, "reservoir_csv"])
    reservoir = pd.read_csv(reservoir_csv)

    assert set(reservoir["time_s"]) == {0.0, 0.04, 0.08}
    assert "reservoir_group_time_s" in reservoir.columns
    assert "original_time_s" in reservoir.columns
    assert float(summary.loc[0, "time_bin_s"]) == 0.1


def test_timebin_sweep_cli_writes_variant_outputs(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = timebin_sweep.main(
        [
            "--candidate-csv",
            f"raw={raw}",
            "--candidate-csv",
            f"translated={translated}",
            "--output-dir",
            str(output),
            "--time-bin-s",
            "0,0.1",
            "--per-source-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
        ]
    )

    assert rc == 0
    summary_path = output / "mmuad_branch_reservoir_timebin_sweep_summary.csv"
    summary_json_path = output / "mmuad_branch_reservoir_timebin_sweep_summary.json"
    assert summary_path.exists()
    assert summary_json_path.exists()
    summary = pd.read_csv(summary_path)
    assert set(summary["time_bin_label"]) == {"timebin_exact", "timebin_0p1s"}
    for reservoir_csv in summary["reservoir_csv"]:
        assert Path(reservoir_csv).exists()
    payload = json.loads(summary_json_path.read_text())
    assert len(payload["rows"]) == 2
