from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    build_oracle_recall_tables,
    build_reservoir_summary,
    load_candidate_inputs,
    main as reservoir_main,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["lidar_360", "lidar_360", "lidar_360", "livox_avia"],
            "track_id": ["raw-good", "calib-bad", "raw-good-1", "calib-bad-1"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "x_m": [0.0, 20.0, 1.0, 18.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "confidence": [0.1, 0.99, 0.2, 0.95],
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


def test_reservoir_preserves_lower_scored_branch_candidates() -> None:
    reservoir = build_candidate_reservoir(
        _candidate_rows(),
        config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=1,
            max_candidates_per_frame=4,
            score_column="ranker_score",
            fallback_score_column="confidence",
        ),
    )

    first_frame = reservoir.loc[reservoir["time_s"] == 0.0]
    assert set(first_frame["candidate_branch"]) == {"raw", "translated"}
    raw_reason = first_frame.loc[first_frame["candidate_branch"] == "raw", "candidate_reservoir_reason"].iloc[0]
    assert "branch:raw" in raw_reason
    translated_reason = first_frame.loc[
        first_frame["candidate_branch"] == "translated",
        "candidate_reservoir_reason",
    ].iloc[0]
    assert "global_top_n" in translated_reason


def test_candidate_reservoir_keeps_per_branch_candidates() -> None:
    rows = _candidate_rows()

    reservoir = build_candidate_reservoir(
        rows,
        top_per_source=0,
        top_per_branch=1,
        global_top_n=1,
    )

    selected = set(reservoir["track_id"])
    assert "raw-good" in selected
    assert "calib-bad" in selected
    assert "raw-good-1" in selected
    assert "calib-bad-1" in selected
    assert "candidate_reservoir_reasons" in reservoir.columns
    assert reservoir["candidate_reservoir_rank"].min() == 1


def test_candidate_reservoir_summary_counts_branches_and_reasons() -> None:
    rows = _candidate_rows()
    reservoir = build_candidate_reservoir(rows, top_per_source=1, top_per_branch=1, global_top_n=1)

    summary = build_reservoir_summary(rows, reservoir)

    assert summary["input_candidate_rows"] == 4
    assert summary["reservoir_candidate_rows"] >= 2
    assert summary["candidate_branch_counts"]["translated"] >= 1
    assert any(key.startswith("branch:") for key in summary["reservoir_reason_counts"])


def test_oracle_recall_tables_report_topk_and_all_candidate_error() -> None:
    reservoir = build_candidate_reservoir(
        _candidate_rows(),
        config=ReservoirConfig(global_top_n=1, per_branch_top_n=1),
    )

    frame_rows, pooled, by_sequence = build_oracle_recall_tables(
        reservoir,
        _truth_rows(),
        top_k_values=(1, 2),
        max_truth_time_delta_s=0.1,
    )

    assert len(frame_rows) == 2
    assert pooled.loc[0, "frame_count"] == 2
    assert pooled.loc[0, "oracle_all_3d_m_mse"] == 0.0
    assert pooled.loc[0, "oracle_top1_3d_m_mse"] > 0.0
    assert pooled.loc[0, "oracle_top2_3d_m_mse"] == 0.0
    assert by_sequence.loc[0, "sequence_id"] == "seqA"


def test_candidate_reservoir_cli_writes_outputs(tmp_path) -> None:
    raw_csv = tmp_path / "raw.csv"
    translated_csv = tmp_path / "translated.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_csv = tmp_path / "summary.csv"
    summary_json = tmp_path / "summary.json"
    by_sequence_csv = tmp_path / "by_sequence.csv"
    frame_csv = tmp_path / "frames.csv"

    rows = _candidate_rows()
    rows.loc[rows["candidate_branch"] == "raw"].drop(columns=["candidate_branch"]).to_csv(
        raw_csv,
        index=False,
    )
    rows.loc[rows["candidate_branch"] == "translated"].drop(
        columns=["candidate_branch"],
    ).to_csv(translated_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = reservoir_main(
        [
            "--candidate",
            f"raw={raw_csv}",
            "--candidate",
            f"translated={translated_csv}",
            "--output-csv",
            str(output_csv),
            "--truth-csv",
            str(truth_csv),
            "--oracle-frame-csv",
            str(frame_csv),
            "--oracle-summary-csv",
            str(summary_csv),
            "--summary-json",
            str(summary_json),
            "--oracle-by-sequence-csv",
            str(by_sequence_csv),
            "--global-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    assert summary_csv.exists()
    assert summary_json.exists()
    assert by_sequence_csv.exists()
    assert frame_csv.exists()
    assert set(pd.read_csv(output_csv)["candidate_branch"]) == {"raw", "translated"}
    assert json.loads(summary_json.read_text(encoding="utf-8"))["reservoir_frame_count"] == 2


def test_candidate_reservoir_cli_accepts_candidate_csv_alias(tmp_path) -> None:
    raw_csv = tmp_path / "raw.csv"
    translated_csv = tmp_path / "translated.csv"
    _candidate_rows().loc[lambda frame: frame["candidate_branch"] == "raw"].to_csv(
        raw_csv,
        index=False,
    )
    _candidate_rows().loc[lambda frame: frame["candidate_branch"] != "raw"].to_csv(
        translated_csv,
        index=False,
    )
    output_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"

    status = reservoir_main(
        [
            "--candidate-csv",
            f"raw={raw_csv}",
            "--candidate-csv",
            f"translated={translated_csv}",
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--top-per-branch",
            "1",
            "--global-top-n",
            "1",
        ]
    )

    assert status == 0
    reservoir = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert set(reservoir["candidate_branch"]) == {"raw", "translated"}
    assert summary["reservoir_frame_count"] == 2


def test_load_candidate_inputs_adds_branch_and_original_coordinates(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    _candidate_rows().drop(columns=["candidate_branch"]).to_csv(candidate_csv, index=False)

    rows = load_candidate_inputs([f"branchA={candidate_csv}"])

    assert set(rows["candidate_branch"]) == {"branchA"}
    assert {"original_x_m", "original_y_m", "original_z_m"}.issubset(rows.columns)
