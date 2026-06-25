from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_branch_reservoir_build.py"
spec = importlib.util.spec_from_file_location("mmuad_branch_reservoir_build", MODULE_PATH)
assert spec is not None and spec.loader is not None
reservoir_build = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reservoir_build
spec.loader.exec_module(reservoir_build)


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


def test_truth_free_reservoir_preserves_per_branch_candidates() -> None:
    reservoir = reservoir_build.build_truth_free_branch_reservoir(
        _candidate_rows(),
        per_source_top_n=0,
        per_branch_top_n=1,
        global_top_n=1,
    )

    assert set(reservoir["candidate_branch"]) == {"raw", "source_translation", "radar"}
    assert set(reservoir["track_id"].astype(str)) == {"raw_bad", "translated", "radar"}
    assert reservoir["reservoir_count_frame"].nunique() == 1
    assert int(reservoir["reservoir_count_frame"].iloc[0]) == 3
    assert any("branch:source_translation" in item for item in reservoir["reservoir_reason"])


def test_truth_free_reservoir_frame_and_branch_summaries() -> None:
    reservoir = reservoir_build.build_truth_free_branch_reservoir(
        _candidate_rows(),
        per_source_top_n=2,
        per_branch_top_n=1,
        global_top_n=1,
    )

    frame_summary = reservoir_build.build_frame_summary(reservoir)
    branch_summary = reservoir_build.build_branch_summary(reservoir)

    assert len(frame_summary) == 1
    assert int(frame_summary.loc[0, "input_candidate_count"]) == 4
    assert int(frame_summary.loc[0, "branch_count"]) == 3
    assert set(branch_summary["candidate_branch"]) == {"raw", "source_translation", "radar"}
    raw = branch_summary.loc[branch_summary["candidate_branch"] == "raw"].iloc[0]
    assert int(raw["candidate_count"]) == 2


def test_truth_free_reservoir_cli_writes_tracker_ready_csv(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = reservoir_build.main(
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
        "mmuad_branch_reservoir_build_provenance.json",
    ):
        assert (output / name).exists()
    reservoir = pd.read_csv(output / "mmuad_branch_reservoir_candidates.csv")
    assert {"sequence_id", "time_s", "x_m", "y_m", "z_m", "candidate_branch"} <= set(
        reservoir.columns
    )
    provenance = json.loads((output / "mmuad_branch_reservoir_build_provenance.json").read_text())
    assert provenance["reservoir_candidate_rows"] == 3
    assert provenance["candidate_inputs"][0]["branch"] == "raw"
