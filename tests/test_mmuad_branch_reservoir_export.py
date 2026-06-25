from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_branch_reservoir_export.py"
spec = importlib.util.spec_from_file_location("mmuad_branch_reservoir_export", MODULE_PATH)
assert spec is not None and spec.loader is not None
reservoir_export = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reservoir_export
spec.loader.exec_module(reservoir_export)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001", "seq001"],
            "time_s": [0.0, 0.0, 0.0, 0.0],
            "source": ["lidar_360", "lidar_360", "livox_avia", "radar"],
            "track_id": ["raw_good", "raw_bad", "translated", "radar"],
            "candidate_branch": ["raw", "raw", "source_translation", "radar"],
            "x_m": [0.2, 20.0, 5.0, 8.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [0.1, 0.99, 0.5, 0.4],
            "confidence": [0.1, 0.99, 0.5, 0.4],
        }
    )


def test_truth_free_reservoir_preserves_low_score_branch_candidate() -> None:
    reservoir, frame_summary, branch_summary = reservoir_export.build_branch_reservoir_export_tables(
        _candidate_rows(),
        config=reservoir_export.ReservoirExportConfig(
            per_source_top_n=0,
            per_branch_top_n=1,
            global_top_n=1,
        ),
    )

    assert set(reservoir["track_id"].astype(str)) == {"raw_bad", "translated", "radar"}
    assert set(reservoir["candidate_branch"].astype(str)) == {
        "raw",
        "source_translation",
        "radar",
    }
    assert set(reservoir["reservoir_reason"].astype(str)) >= {"branch_top", "global_top;branch_top"}
    assert int(frame_summary.loc[0, "candidate_count"]) == 4
    assert int(frame_summary.loc[0, "retained_count"]) == 3
    raw_summary = branch_summary.loc[branch_summary["candidate_branch"] == "raw"].iloc[0]
    assert int(raw_summary["input_count"]) == 2
    assert int(raw_summary["retained_count"]) == 1


def test_truth_free_reservoir_time_bin_groups_nearby_times() -> None:
    rows = _candidate_rows().copy()
    rows.loc[0, "time_s"] = 0.01
    rows.loc[1, "time_s"] = 0.02
    rows.loc[2, "time_s"] = 0.03
    rows.loc[3, "time_s"] = 0.04

    reservoir, frame_summary, _ = reservoir_export.build_branch_reservoir_export_tables(
        rows,
        config=reservoir_export.ReservoirExportConfig(
            per_source_top_n=0,
            per_branch_top_n=1,
            global_top_n=1,
            candidate_time_bin_s=0.1,
        ),
    )

    assert len(frame_summary) == 1
    assert int(frame_summary.loc[0, "candidate_count"]) == 4
    assert int(frame_summary.loc[0, "retained_count"]) == len(reservoir)


def test_truth_free_reservoir_cli_writes_artifacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = reservoir_export.main(
        [
            "--candidate-csv",
            f"raw={raw}",
            "--candidate-csv",
            f"translated={translated}",
            "--output-dir",
            str(output),
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
        ]
    )

    assert rc == 0
    for name in (
        "mmuad_branch_reservoir_candidates.csv",
        "mmuad_branch_reservoir_frame_summary.csv",
        "mmuad_branch_reservoir_branch_summary.csv",
        "mmuad_branch_reservoir_export_provenance.json",
    ):
        assert (output / name).exists()
    reservoir = pd.read_csv(output / "mmuad_branch_reservoir_candidates.csv")
    assert set(reservoir["candidate_branch"].astype(str)) == {"raw", "translated"}
    provenance = json.loads((output / "mmuad_branch_reservoir_export_provenance.json").read_text())
    assert provenance["truth_free"] is True
    assert provenance["config"]["global_top_n"] == 1
