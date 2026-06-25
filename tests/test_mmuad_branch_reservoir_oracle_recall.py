from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_branch_reservoir_oracle_recall.py"
spec = importlib.util.spec_from_file_location("mmuad_branch_reservoir_oracle_recall", MODULE_PATH)
assert spec is not None and spec.loader is not None
reservoir_oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reservoir_oracle
spec.loader.exec_module(reservoir_oracle)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


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


def test_branch_reservoir_oracle_reports_topk_recall_gap() -> None:
    frame_rows, pooled, _, reservoir = reservoir_oracle.build_branch_reservoir_oracle_tables(
        _candidate_rows(),
        _truth_rows(),
        max_time_delta_s=0.1,
        top_k_values=(1, 3),
        per_source_top_n=2,
        per_branch_top_n=1,
        global_top_n=1,
    )

    top1 = pooled.loc[pooled["top_k"] == "1"].iloc[0]
    top3 = pooled.loc[pooled["top_k"] == "3"].iloc[0]
    all_row = pooled.loc[pooled["top_k"] == "all"].iloc[0]

    assert float(top1["oracle_mse_m2"]) == 400.0
    assert float(top3["oracle_mse_m2"]) == 25.0
    assert float(all_row["oracle_mse_m2"]) == 0.04000000000000001
    assert "raw_good" in set(reservoir["track_id"])
    assert len(frame_rows) == 3


def test_branch_reservoir_oracle_cli_writes_artifacts(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    _truth_rows().to_csv(truth, index=False)
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].to_csv(raw, index=False)
    rows.loc[rows["candidate_branch"] != "raw"].to_csv(translated, index=False)

    rc = reservoir_oracle.main(
        [
            "--truth-file",
            str(truth),
            "--candidate-csv",
            f"raw={raw}",
            "--candidate-csv",
            f"translated={translated}",
            "--output-dir",
            str(output),
            "--top-k",
            "1,3",
            "--per-source-top-n",
            "2",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
            "--write-reservoir-candidates",
        ]
    )

    assert rc == 0
    for name in (
        "mmuad_branch_reservoir_oracle_frame_rows.csv",
        "mmuad_branch_reservoir_oracle_pooled.csv",
        "mmuad_branch_reservoir_oracle_by_sequence.csv",
        "mmuad_branch_reservoir_candidates.csv",
        "mmuad_branch_reservoir_oracle_provenance.json",
    ):
        assert (output / name).exists()
    provenance = json.loads((output / "mmuad_branch_reservoir_oracle_provenance.json").read_text())
    assert provenance["candidate_inputs"][0]["branch"] == "raw"
