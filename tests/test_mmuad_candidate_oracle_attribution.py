from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_oracle_attribution import (
    build_candidate_oracle_attribution_tables,
    main as attribution_main,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["lidar_360", "livox_avia", "lidar_360", "livox_avia"],
            "track_id": ["raw-0", "translated-0", "raw-1", "translated-1"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "x_m": [0.0, 20.0, 1.0, 18.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "candidate_reservoir_score": [0.10, 0.90, 0.20, 0.95],
            "ranker_score": [0.10, 0.90, 0.20, 0.95],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )


def test_oracle_attribution_reports_winning_branch_and_buried_rank() -> None:
    frame_rows, pooled, by_branch, by_source = build_candidate_oracle_attribution_tables(
        _candidate_rows(),
        _truth_rows(),
        top_k_values=(1, 2),
        max_truth_time_delta_s=0.1,
    )

    assert len(frame_rows) == 2
    assert set(frame_rows["oracle_all_candidate_branch"]) == {"raw"}
    assert frame_rows["oracle_all_rank"].tolist() == [2, 2]
    assert pooled.loc[0, "oracle_all_3d_m_mse"] == 0.0
    assert pooled.loc[0, "oracle_top1_3d_m_mse"] > 0.0
    assert pooled.loc[0, "oracle_top2_3d_m_mse"] == 0.0
    assert pooled.loc[0, "oracle_in_top1_fraction"] == 0.0
    assert pooled.loc[0, "oracle_in_top2_fraction"] == 1.0
    assert by_branch.loc[0, "candidate_branch"] == "raw"
    assert by_branch.loc[0, "winning_frames"] == 2
    assert by_source.loc[0, "source"] == "lidar_360"


def test_oracle_attribution_cli_writes_tables(tmp_path) -> None:
    raw_csv = tmp_path / "raw.csv"
    translated_csv = tmp_path / "translated.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].drop(columns=["candidate_branch"]).to_csv(
        raw_csv,
        index=False,
    )
    rows.loc[rows["candidate_branch"] == "translated"].drop(columns=["candidate_branch"]).to_csv(
        translated_csv,
        index=False,
    )
    _truth_rows().to_csv(truth_csv, index=False)

    rc = attribution_main(
        [
            "--candidate",
            f"raw={raw_csv}",
            "--candidate",
            f"translated={translated_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
            "--top-k",
            "2",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert rc == 0
    frame_rows = pd.read_csv(output_dir / "mmuad_candidate_oracle_attribution_frames.csv")
    pooled = pd.read_csv(output_dir / "mmuad_candidate_oracle_attribution_pooled.csv")
    summary = json.loads(
        (output_dir / "mmuad_candidate_oracle_attribution_summary.json").read_text(
            encoding="utf-8",
        ),
    )
    assert set(frame_rows["oracle_all_candidate_branch"]) == {"raw"}
    assert pooled.loc[0, "oracle_top2_3d_m_mse"] == 0.0
    assert "oracle_top3_3d_m" not in frame_rows.columns
    assert "oracle_top3_3d_m_mse" not in pooled.columns
    assert summary["by_branch"][0]["candidate_branch"] == "raw"
