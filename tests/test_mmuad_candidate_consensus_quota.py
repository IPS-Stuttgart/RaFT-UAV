from __future__ import annotations

import json

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_consensus_quota import (
    build_consensus_quota_reservoir,
    consensus_quota_summary,
    main as consensus_quota_main,
)
from raft_uav.mmuad.candidate_reservoir import ReservoirConfig


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0] * 4,
            "source": ["lidar_360", "lidar_360", "livox_avia", "radar"],
            "candidate_branch": ["raw", "raw", "raw", "raw"],
            "track_id": ["isolated-high", "supported-low", "support", "isolated-radar"],
            "x_m": [50.0, 0.0, 0.2, 100.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "ranker_score": [0.99, 0.10, 0.05, 0.90],
            "mmuad_calibration_origin_row": ["a", "b", "c", "d"],
        }
    )


def _origin_sibling_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0] * 4,
            "source": ["lidar_360", "lidar_360", "livox_avia", "radar"],
            "candidate_branch": ["raw", "translated", "raw", "raw"],
            "track_id": ["raw-copy", "translated-copy", "avia", "radar"],
            "x_m": [0.0, 0.1, 0.05, 20.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "ranker_score": [0.8, 0.9, 0.7, 0.95],
            "mmuad_calibration_origin_row": ["shared", "shared", "avia", "radar"],
        }
    )


def _config(*, max_candidates_per_frame: int = 2) -> ReservoirConfig:
    return ReservoirConfig(
        global_top_n=1,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=max_candidates_per_frame,
        score_column="ranker_score",
        fallback_score_column="confidence",
    )


def test_consensus_quota_recovers_supported_low_score_candidate() -> None:
    reservoir = build_consensus_quota_reservoir(
        _candidate_rows(),
        reservoir_config=_config(),
        consensus_top_n=1,
        max_nearest_distance_m=1.0,
    ).rows

    assert set(reservoir["track_id"]) == {"isolated-high", "supported-low"}
    supported = reservoir.loc[reservoir["track_id"] == "supported-low"].iloc[0]
    assert bool(supported["candidate_consensus_quota_selected"])
    assert supported["candidate_consensus_quota_rank"] == 1
    assert "consensus:cross_source" in supported["candidate_reservoir_reason"]
    assert supported["branch_consensus_nearest_cross_source_distance_m"] == pytest.approx(0.2)


def test_consensus_quota_is_mandatory_under_hard_frame_cap() -> None:
    reservoir = build_consensus_quota_reservoir(
        _candidate_rows(),
        reservoir_config=_config(max_candidates_per_frame=1),
        consensus_top_n=1,
        max_nearest_distance_m=1.0,
    ).rows

    assert reservoir["track_id"].tolist() == ["supported-low"]
    assert bool(reservoir.iloc[0]["candidate_reservoir_protected"])


def test_consensus_quota_limits_duplicate_coordinate_branches_per_origin() -> None:
    reservoir = build_consensus_quota_reservoir(
        _origin_sibling_rows(),
        reservoir_config=_config(max_candidates_per_frame=3),
        consensus_top_n=2,
        max_per_origin=1,
        max_nearest_distance_m=1.0,
    ).rows

    quota = reservoir.loc[reservoir["candidate_consensus_quota_selected"]].copy()
    assert len(quota) == 2
    assert quota["mmuad_calibration_origin_row"].nunique() == 2
    assert (quota["mmuad_calibration_origin_row"] == "shared").sum() == 1


def test_zero_consensus_quota_recovers_base_reservoir_selection() -> None:
    reservoir = build_consensus_quota_reservoir(
        _candidate_rows(),
        reservoir_config=_config(),
        consensus_top_n=0,
    ).rows

    assert reservoir["track_id"].tolist() == ["isolated-high"]
    assert not reservoir["candidate_consensus_quota_selected"].any()


def test_consensus_summary_reports_selected_rows() -> None:
    rows = _candidate_rows()
    reservoir = build_consensus_quota_reservoir(
        rows,
        reservoir_config=_config(),
        consensus_top_n=1,
        max_nearest_distance_m=1.0,
    )

    summary = consensus_quota_summary(rows, reservoir)

    assert summary["consensus_quota_selected_rows"] == 1
    assert summary["consensus_quota_selected_frame_count"] == 1
    assert summary["consensus_supported_output_rows"] >= 1


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("consensus_top_n", -1),
        ("min_neighbor_count", -1),
        ("min_unique_source_count", -1),
        ("max_per_origin", 0),
        ("max_per_source", -1),
        ("max_nearest_distance_m", float("nan")),
    ],
)
def test_consensus_quota_rejects_invalid_controls(name: str, value: float) -> None:
    kwargs = {name: value}
    with pytest.raises(ValueError):
        build_consensus_quota_reservoir(
            _candidate_rows(),
            reservoir_config=_config(),
            **kwargs,
        )


def test_consensus_quota_cli_writes_summary_and_oracle_tables(tmp_path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    oracle_frame_csv = tmp_path / "oracle_frames.csv"
    oracle_summary_csv = tmp_path / "oracle_summary.csv"
    oracle_by_sequence_csv = tmp_path / "oracle_by_sequence.csv"
    _candidate_rows().to_csv(candidates_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(truth_csv, index=False)

    status = consensus_quota_main(
        [
            "--candidates-csv",
            str(candidates_csv),
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
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "2",
            "--consensus-top-n",
            "1",
            "--max-nearest-distance-m",
            "1",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    assert oracle_frame_csv.exists()
    assert oracle_summary_csv.exists()
    assert oracle_by_sequence_csv.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["consensus_top_n"] == 1
    assert summary["consensus_quota_selected_rows"] == 1
    oracle = pd.read_csv(oracle_summary_csv)
    assert oracle.loc[0, "oracle_all_3d_m_mse"] == pytest.approx(0.0)
