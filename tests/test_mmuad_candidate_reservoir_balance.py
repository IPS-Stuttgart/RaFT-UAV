from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_balance import (
    ReservoirBalanceConfig,
    balance_candidate_reservoir,
    build_balance_summary,
    main as reservoir_balance_main,
)


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 6,
            "time_s": [1.0] * 6,
            "source": [
                "lidar_360",
                "lidar_360",
                "lidar_360",
                "livox_avia",
                "livox_avia",
                "radar_enhance_pcl",
            ],
            "candidate_branch": [
                "translated",
                "translated",
                "translated",
                "raw",
                "dynamic",
                "radar",
            ],
            "track_id": ["t0", "t1", "t2", "raw", "dynamic", "radar"],
            "x_m": [20.0, 21.0, 22.0, 0.0, 0.5, 2.0],
            "y_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0] * 6,
            "candidate_reservoir_score": [0.99, 0.98, 0.97, 0.10, 0.05, 0.01],
        }
    )


def test_balance_preserves_low_score_branch_representatives() -> None:
    balanced = balance_candidate_reservoir(
        _candidate_rows(),
        config=ReservoirBalanceConfig(
            max_candidates_per_frame=4,
            min_per_branch=1,
            min_per_source=0,
        ),
    )

    assert len(balanced) == 4
    assert set(balanced["candidate_branch"]) == {
        "translated",
        "raw",
        "dynamic",
        "radar",
    }
    assert balanced["candidate_reservoir_balance_protected"].all()
    assert balanced["candidate_reservoir_balanced_rank"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_balance_preserves_source_representatives_before_global_fill() -> None:
    balanced = balance_candidate_reservoir(
        _candidate_rows(),
        config=ReservoirBalanceConfig(
            max_candidates_per_frame=4,
            min_per_branch=0,
            min_per_source=1,
        ),
    )

    assert set(balanced["source"]) == {
        "lidar_360",
        "livox_avia",
        "radar_enhance_pcl",
    }
    assert "t1" in set(balanced["track_id"])
    assert any("global_fill" in value for value in balanced["candidate_reservoir_balance_reason"])


def test_balance_tight_cap_is_deterministic_when_all_branches_cannot_fit() -> None:
    balanced = balance_candidate_reservoir(
        _candidate_rows(),
        config=ReservoirBalanceConfig(
            max_candidates_per_frame=2,
            min_per_branch=1,
            min_per_source=0,
        ),
    )

    assert balanced["track_id"].tolist() == ["t0", "raw"]
    assert balanced["candidate_reservoir_balanced_rank"].tolist() == [1.0, 2.0]


def test_balance_summary_reports_coverage_loss() -> None:
    rows = _candidate_rows()
    balanced = balance_candidate_reservoir(
        rows,
        config=ReservoirBalanceConfig(
            max_candidates_per_frame=4,
            min_per_branch=1,
            min_per_source=1,
        ),
    )

    summary = build_balance_summary(rows, balanced)

    assert summary["input_candidate_rows"] == 6
    assert summary["balanced_candidate_rows"] == 4
    assert summary["frames_with_branch_coverage_loss"] == 0
    assert summary["frames_with_source_coverage_loss"] == 0
    assert summary["balanced_candidates_per_frame_max"] == 4


def test_balance_cli_writes_csv_and_summary(tmp_path) -> None:
    input_csv = tmp_path / "reservoir.csv"
    output_csv = tmp_path / "balanced.csv"
    summary_json = tmp_path / "summary.json"
    _candidate_rows().to_csv(input_csv, index=False)

    status = reservoir_balance_main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--max-candidates-per-frame",
            "4",
            "--min-per-branch",
            "1",
            "--min-per-source",
            "0",
        ]
    )

    assert status == 0
    assert len(pd.read_csv(output_csv)) == 4
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["balanced_candidate_rows"] == 4
    assert summary["frames_with_branch_coverage_loss"] == 0
