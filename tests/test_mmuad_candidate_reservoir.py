from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    build_candidate_reservoir,
    build_reservoir_summary,
    load_candidate_inputs,
    main as reservoir_main,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 5,
            "time_s": [1.0] * 5,
            "source": ["lidar_360", "lidar_360", "livox_avia", "livox_avia", "radar"],
            "track_id": ["raw_bad", "raw_good", "translated", "dynamic", "radar"],
            "candidate_branch": ["raw", "raw", "source_translation", "dynamic", "radar"],
            "x_m": [20.0, 0.0, 0.2, 0.4, 5.0],
            "y_m": [0.0, 0.0, 0.2, 0.0, 5.0],
            "z_m": [0.0, 0.0, 0.0, 0.0, 0.0],
            "ranker_score": [0.99, 0.10, 0.20, 0.15, 0.05],
            "confidence": [0.99, 0.10, 0.20, 0.15, 0.05],
        }
    )


def test_candidate_reservoir_keeps_per_branch_candidates() -> None:
    rows = _candidate_rows()

    reservoir = build_candidate_reservoir(
        rows,
        top_per_source=0,
        top_per_branch=1,
        global_top_n=1,
    )

    selected = set(reservoir["track_id"])
    assert "raw_bad" in selected  # global top-1 by score
    assert "translated" in selected  # preserved by branch top-1 despite lower score
    assert "dynamic" in selected
    assert "radar" in selected
    assert "candidate_reservoir_reasons" in reservoir.columns
    assert reservoir["candidate_reservoir_rank"].min() == 1


def test_candidate_reservoir_summary_counts_branches_and_reasons() -> None:
    rows = _candidate_rows()
    reservoir = build_candidate_reservoir(rows, top_per_source=1, top_per_branch=1, global_top_n=1)

    summary = build_reservoir_summary(rows, reservoir)

    assert summary["input_candidate_rows"] == 5
    assert summary["reservoir_candidate_rows"] >= 3
    assert summary["candidate_branch_counts"]["source_translation"] == 1
    assert any(key.startswith("branch:") for key in summary["reservoir_reason_counts"])


def test_candidate_reservoir_cli_writes_outputs(tmp_path) -> None:
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

    rc = reservoir_main(
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

    assert rc == 0
    reservoir = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert set(reservoir["candidate_branch"]) >= {"raw", "source_translation", "dynamic"}
    assert summary["reservoir_frame_count"] == 1


def test_load_candidate_inputs_adds_branch_and_original_coordinates(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    _candidate_rows().drop(columns=["candidate_branch"]).to_csv(candidate_csv, index=False)

    rows = load_candidate_inputs([f"branchA={candidate_csv}"])

    assert set(rows["candidate_branch"]) == {"branchA"}
    assert {"original_x_m", "original_y_m", "original_z_m"}.issubset(rows.columns)
