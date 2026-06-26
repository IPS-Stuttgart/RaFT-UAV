from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_build_branch_reservoir.py"
spec = importlib.util.spec_from_file_location("mmuad_build_branch_reservoir", MODULE_PATH)
assert spec is not None and spec.loader is not None
reservoir_builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reservoir_builder
spec.loader.exec_module(reservoir_builder)


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


def test_truth_free_reservoir_preserves_branch_low_score_candidate() -> None:
    reservoir = reservoir_builder.build_truth_free_branch_reservoir(
        _candidate_rows(),
        per_source_top_n=2,
        per_branch_top_n=1,
        global_top_n=1,
    )

    assert "raw_good" in set(reservoir["track_id"])
    assert "raw_bad" in set(reservoir["track_id"])
    assert "translated" in set(reservoir["track_id"])
    raw_good = reservoir.loc[reservoir["track_id"] == "raw_good"].iloc[0]
    assert "source:lidar_360" in raw_good["reservoir_selected_by"]


def test_truth_free_reservoir_can_apply_frame_cap() -> None:
    reservoir = reservoir_builder.build_truth_free_branch_reservoir(
        _candidate_rows(),
        per_source_top_n=2,
        per_branch_top_n=1,
        global_top_n=1,
        max_candidates_per_frame=2,
    )

    assert len(reservoir) == 2
    assert list(reservoir["track_id"]) == ["raw_bad", "translated"]


def test_frame_and_branch_summaries_count_retention() -> None:
    candidates = _candidate_rows()
    reservoir = reservoir_builder.build_truth_free_branch_reservoir(
        candidates,
        per_source_top_n=2,
        per_branch_top_n=1,
        global_top_n=1,
    )

    frame_summary = reservoir_builder.build_frame_summary(candidates, reservoir)
    branch_summary = reservoir_builder.build_branch_summary(candidates, reservoir)

    assert int(frame_summary.loc[0, "candidate_count"]) == 4
    assert int(frame_summary.loc[0, "reservoir_count"]) == len(reservoir)
    raw_summary = branch_summary.loc[branch_summary["candidate_branch"] == "raw"].iloc[0]
    assert int(raw_summary["candidate_count"]) == 2
    assert int(raw_summary["reservoir_count"]) == 2


def test_truth_free_reservoir_cli_writes_artifacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = reservoir_builder.main(
        [
            "--candidate-csv",
            f"raw={raw}",
            "--candidate-csv",
            f"translated={translated}",
            "--output-dir",
            str(output),
            "--per-source-top-n",
            "2",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
        ]
    )

    assert rc == 0
    for name in (
        "mmuad_truth_free_branch_reservoir_candidates.csv",
        "mmuad_truth_free_branch_reservoir_frame_summary.csv",
        "mmuad_truth_free_branch_reservoir_branch_summary.csv",
        "mmuad_truth_free_branch_reservoir_provenance.json",
    ):
        assert (output / name).exists()
    reservoir = pd.read_csv(output / "mmuad_truth_free_branch_reservoir_candidates.csv")
    assert "reservoir_selected_by" in reservoir.columns
    provenance = json.loads((output / "mmuad_truth_free_branch_reservoir_provenance.json").read_text())
    assert provenance["candidate_inputs"][0]["branch"] == "raw"
