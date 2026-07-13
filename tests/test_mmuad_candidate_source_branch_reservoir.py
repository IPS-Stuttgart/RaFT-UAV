from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_source_branch_reservoir import (
    build_source_branch_reservoir,
    main as source_branch_main,
    source_branch_reservoir_summary,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0] * 4,
            "source": ["lidar_360", "lidar_360", "livox_avia", "livox_avia"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "track_id": ["lidar-raw", "lidar-translated", "livox-raw", "livox-translated"],
            "x_m": [0.0, 1.0, 2.0, 3.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "ranker_score": [0.90, 0.10, 0.80, 0.70],
        }
    )


def _diversity_candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": [0.0] * 3,
            "source": ["lidar_360"] * 3,
            "candidate_branch": ["raw"] * 3,
            "track_id": ["best", "near-duplicate", "far-alternative"],
            "x_m": [0.0, 0.1, 20.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
            "ranker_score": [1.00, 0.99, 0.98],
        }
    )


def _config(*, max_candidates_per_frame: int = 8) -> ReservoirConfig:
    return ReservoirConfig(
        global_top_n=1,
        per_source_top_n=1,
        per_branch_top_n=1,
        max_candidates_per_frame=max_candidates_per_frame,
        score_column="ranker_score",
        fallback_score_column="confidence",
    )


def _source_branch_only_config() -> ReservoirConfig:
    return ReservoirConfig(
        global_top_n=0,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=2,
        score_column="ranker_score",
        fallback_score_column="confidence",
    )


def test_source_branch_quota_recovers_intersection_missed_by_independent_quotas() -> None:
    reservoir = build_source_branch_reservoir(
        _candidate_rows(),
        reservoir_config=_config(),
        per_source_branch_top_n=1,
    ).rows

    assert set(reservoir["track_id"]) == {
        "lidar-raw",
        "lidar-translated",
        "livox-raw",
        "livox-translated",
    }
    recovered = reservoir.loc[reservoir["track_id"] == "lidar-translated"].iloc[0]
    assert "source_branch:lidar_360|translated" in recovered["candidate_reservoir_reason"]
    assert bool(recovered["candidate_source_branch_quota_selected"])


def test_disabling_source_branch_quota_matches_independent_reservoir_selection() -> None:
    reservoir = build_source_branch_reservoir(
        _candidate_rows(),
        reservoir_config=_config(),
        per_source_branch_top_n=0,
    ).rows

    assert set(reservoir["track_id"]) == {
        "lidar-raw",
        "livox-raw",
        "livox-translated",
    }
    assert not reservoir["candidate_source_branch_quota_selected"].any()


def test_source_branch_quota_respects_final_frame_cap() -> None:
    reservoir = build_source_branch_reservoir(
        _candidate_rows(),
        reservoir_config=_config(max_candidates_per_frame=3),
        per_source_branch_top_n=1,
    ).rows

    assert len(reservoir) == 3
    assert reservoir["candidate_reservoir_rank"].tolist() == [1.0, 2.0, 3.0]
    assert reservoir["candidate_reservoir_protected"].all()


def test_source_branch_diversity_replaces_near_duplicate_with_alternative() -> None:
    reservoir = build_source_branch_reservoir(
        _diversity_candidate_rows(),
        reservoir_config=_source_branch_only_config(),
        per_source_branch_top_n=2,
        source_branch_diversity_weight=2.0,
        source_branch_diversity_scale_m=1.0,
        source_branch_distance_cap_m=50.0,
    ).rows

    assert set(reservoir["track_id"]) == {"best", "far-alternative"}
    alternative = reservoir.loc[reservoir["track_id"] == "far-alternative"].iloc[0]
    assert alternative["candidate_source_branch_min_distance_m"] > 19.0
    assert alternative["candidate_source_branch_diversity_term"] > 0.99
    assert bool(alternative["candidate_source_branch_diversity_selected"])


def test_zero_source_branch_diversity_weight_preserves_score_order() -> None:
    reservoir = build_source_branch_reservoir(
        _diversity_candidate_rows(),
        reservoir_config=_source_branch_only_config(),
        per_source_branch_top_n=2,
        source_branch_diversity_weight=0.0,
    ).rows

    assert set(reservoir["track_id"]) == {"best", "near-duplicate"}
    assert not reservoir["candidate_source_branch_diversity_selected"].any()


def test_source_branch_summary_reports_cell_recall() -> None:
    rows = _candidate_rows()
    reservoir = build_source_branch_reservoir(
        rows,
        reservoir_config=_config(),
        per_source_branch_top_n=1,
    )

    summary = source_branch_reservoir_summary(rows, reservoir)

    assert summary["input_source_branch_cells"] == 4
    assert summary["retained_source_branch_cells"] == 4
    assert summary["source_branch_cell_recall"] == 1.0
    assert summary["source_branch_quota_selected_rows"] == 4


def test_source_branch_summary_parses_csv_boolean_flags() -> None:
    rows = _candidate_rows()
    reservoir = build_source_branch_reservoir(
        rows,
        reservoir_config=_config(),
        per_source_branch_top_n=1,
    ).rows.copy()
    reservoir["candidate_source_branch_quota_selected"] = [
        "True",
        "False",
        "1",
        "0",
    ]
    reservoir["candidate_source_branch_diversity_selected"] = [
        "yes",
        "no",
        "null",
        "2",
    ]

    summary = source_branch_reservoir_summary(rows, reservoir)

    assert summary["source_branch_quota_selected_rows"] == 2
    assert summary["source_branch_diversity_selected_rows"] == 2


def test_source_branch_cli_writes_reservoir_summary_and_oracle_tables(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    oracle_frame_csv = tmp_path / "oracle_frames.csv"
    oracle_summary_csv = tmp_path / "oracle_summary.csv"
    oracle_by_sequence_csv = tmp_path / "oracle_by_sequence.csv"
    _candidate_rows().to_csv(candidate_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [1.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(truth_csv, index=False)

    status = source_branch_main(
        [
            "--candidates-csv",
            str(candidate_csv),
            "--output-reservoir-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--truth-csv",
            str(truth_csv),
            "--oracle-frame-csv",
            str(oracle_frame_csv),
            "--oracle-summary-csv",
            str(oracle_summary_csv),
            "--oracle-by-sequence-csv",
            str(oracle_by_sequence_csv),
            "--score-column",
            "ranker_score",
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--per-source-branch-top-n",
            "1",
            "--source-branch-diversity-weight",
            "0.5",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    assert oracle_frame_csv.exists()
    assert oracle_summary_csv.exists()
    assert oracle_by_sequence_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["per_source_branch_top_n"] == 1
    assert summary["source_branch_diversity_weight"] == 0.5
    assert summary["source_branch_cell_recall"] == 1.0
    oracle = pd.read_csv(oracle_summary_csv)
    assert oracle.loc[0, "oracle_all_3d_m_mse"] == 0.0
