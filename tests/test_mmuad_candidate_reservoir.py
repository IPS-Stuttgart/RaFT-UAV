from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_reservoir import build_branch_reservoir
from raft_uav.mmuad.candidate_reservoir import build_topk_oracle_recall
from raft_uav.mmuad.candidate_reservoir import load_branch_candidate_specs
from raft_uav.mmuad.candidate_reservoir import main as reservoir_main
from raft_uav.mmuad.candidate_reservoir import summarize_oracle_recall
from raft_uav.mmuad.candidate_reservoir import tag_candidate_branch


def _truth_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001"],
            "time_s": [0.0, 0.0],
            "source": ["lidar_360", "lidar_360"],
            "track_id": ["raw_far", "translated_near"],
            "x_m": [10.0, 0.1],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
            "confidence": [0.99, 0.01],
        }
    )


def test_branch_reservoir_keeps_low_score_branch_candidate() -> None:
    raw = tag_candidate_branch(_candidate_frame().iloc[[0]], branch="raw")
    translated = tag_candidate_branch(_candidate_frame().iloc[[1]], branch="source_translation")
    candidates = pd.concat([raw, translated], ignore_index=True)

    reservoir = build_branch_reservoir(
        candidates,
        config=ReservoirConfig(per_source_top_n=0, per_branch_top_n=1, global_top_n=1),
    )

    assert set(reservoir["candidate_branch"]) == {"raw", "source_translation"}
    assert set(reservoir["track_id"].astype(str)) == {"raw_far", "translated_near"}


def test_topk_oracle_recall_shows_branch_recall_gain() -> None:
    raw = tag_candidate_branch(_candidate_frame().iloc[[0]], branch="raw")
    translated = tag_candidate_branch(_candidate_frame().iloc[[1]], branch="source_translation")
    candidates = pd.concat([raw, translated], ignore_index=True)
    reservoir = build_branch_reservoir(
        candidates,
        config=ReservoirConfig(per_source_top_n=0, per_branch_top_n=1, global_top_n=1),
    )

    rows = build_topk_oracle_recall(
        reservoir,
        _truth_frame(),
        top_k=(1, 2),
        max_time_delta_s=0.1,
    )
    summary = summarize_oracle_recall(rows)

    top1 = summary.loc[summary["oracle_k"] == "top1"].iloc[0]
    top2 = summary.loc[summary["oracle_k"] == "top2"].iloc[0]
    assert float(top1["oracle_mse"]) == 100.0
    assert float(top2["oracle_mse"]) == 0.010000000000000002


def test_branch_reservoir_cli_writes_artifacts(tmp_path: Path) -> None:
    truth = tmp_path / "truth.csv"
    raw = tmp_path / "raw.csv"
    translated = tmp_path / "translated.csv"
    output = tmp_path / "out"
    _truth_frame().to_csv(truth, index=False)
    _candidate_frame().iloc[[0]].to_csv(raw, index=False)
    _candidate_frame().iloc[[1]].to_csv(translated, index=False)

    status = reservoir_main(
        [
            "--truth-file",
            str(truth),
            "--candidate-csv",
            f"raw={raw}",
            "--candidate-csv",
            f"source_translation={translated}",
            "--output-dir",
            str(output),
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
            "--top-k",
            "1,2",
            "--max-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    assert (output / "mmuad_branch_reservoir_candidates.csv").exists()
    assert (output / "mmuad_branch_reservoir_oracle_rows.csv").exists()
    assert (output / "mmuad_branch_reservoir_oracle_pooled.csv").exists()
    assert (output / "mmuad_branch_reservoir_provenance.json").exists()
    reservoir = pd.read_csv(output / "mmuad_branch_reservoir_candidates.csv")
    assert set(reservoir["candidate_branch"]) == {"raw", "source_translation"}


def test_load_branch_candidate_specs_accepts_branch_prefix(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    _candidate_frame().to_csv(path, index=False)

    rows = load_branch_candidate_specs([f"my_branch={path}"])

    assert set(rows["candidate_branch"]) == {"my_branch"}
    assert set(rows["candidate_branch_file"]) == {str(path)}
