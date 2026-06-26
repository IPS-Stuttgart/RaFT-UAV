from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_temporal_consensus import (
    TemporalConsensusConfig,
    add_temporal_candidate_consensus,
)
from raft_uav.mmuad.candidate_temporal_consensus_assignment import (
    add_assignment_temporal_candidate_consensus,
    assignment_temporal_consensus_summary,
    main as assignment_main,
)
from raft_uav.mmuad.schema import CandidateFrame


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 1.0, 2.0],
            "source": [
                "livox_avia",
                "lidar_360",
                "radar_enhance_pcl",
                "livox_avia",
            ],
            "candidate_branch": ["dynamic", "raw", "translated", "dynamic"],
            "track_id": ["previous", "smooth", "duplicate", "next"],
            "x_m": [0.0, 1.0, 3.5, 2.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "confidence": [0.5, 0.1, 0.99, 0.5],
            "ranker_score": [0.5, 0.1, 0.99, 0.5],
        }
    )


def _config() -> TemporalConsensusConfig:
    return TemporalConsensusConfig(
        max_time_gap_s=1.1,
        max_speed_mps=10.0,
        distance_scale_m=2.0,
        acceleration_scale_mps2=5.0,
    )


def test_one_to_one_assignment_prevents_neighbor_reuse() -> None:
    nearest = add_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
    ).rows
    assigned = add_assignment_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
        assignment_mode="one-to-one",
    ).rows

    nearest_middle = nearest.loc[nearest["time_s"] == 1.0].set_index("track_id")
    assigned_middle = assigned.loc[assigned["time_s"] == 1.0].set_index("track_id")

    assert nearest_middle.loc[
        "duplicate",
        "candidate_reservoir_temporal_bidirectional",
    ] == pytest.approx(1.0)
    assert assigned_middle.loc[
        "smooth",
        "candidate_reservoir_temporal_bidirectional",
    ] == pytest.approx(1.0)
    assert assigned_middle.loc[
        "duplicate",
        "candidate_reservoir_temporal_bidirectional",
    ] == pytest.approx(0.0)
    assert pd.isna(
        assigned_middle.loc[
            "duplicate",
            "candidate_reservoir_temporal_backward_distance_m",
        ]
    )
    assert pd.isna(
        assigned_middle.loc[
            "duplicate",
            "candidate_reservoir_temporal_forward_distance_m",
        ]
    )
    assert assigned_middle.loc[
        "smooth",
        "candidate_temporal_backward_track_id",
    ] == "previous"
    assert assigned_middle.loc[
        "smooth",
        "candidate_temporal_forward_track_id",
    ] == "next"


def test_nearest_assignment_mode_matches_existing_temporal_scores() -> None:
    expected = add_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
    ).rows
    actual = add_assignment_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
        assignment_mode="nearest",
    ).rows

    pd.testing.assert_series_equal(
        actual["candidate_temporal_consensus_score"],
        expected["candidate_temporal_consensus_score"],
        check_names=False,
    )
    assert actual["candidate_temporal_assignment_mode"].eq("nearest").all()
    assert (
        actual["candidate_reservoir_temporal_backward_assignment_matched"].sum()
        == pytest.approx(3.0)
    )


def test_assignment_summary_reports_matched_counts() -> None:
    assigned = add_assignment_temporal_candidate_consensus(
        CandidateFrame(_candidate_rows()),
        config=_config(),
    )
    summary = assignment_temporal_consensus_summary(assigned)

    assert summary["assignment_mode_counts"] == {"one-to-one": 4}
    assert summary["backward_assignment_matched_count"] == 2
    assert summary["forward_assignment_matched_count"] == 2
    assert summary["bidirectional_supported_count"] == 1


def test_assignment_cli_accepts_train_selected_config(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    config_json = tmp_path / "selected_config.json"
    output_csv = tmp_path / "assigned.csv"
    summary_json = tmp_path / "assigned_summary.json"
    _candidate_rows().to_csv(candidates_csv, index=False)
    config_json.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "selection_protocol": "unit-test",
                "selected_config_id": "temporal_0001",
                "temporal_consensus_config": asdict(_config()),
            }
        ),
        encoding="utf-8",
    )

    status = assignment_main(
        [
            "--candidate-csv",
            str(candidates_csv),
            "--config-json",
            str(config_json),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--assignment-mode",
            "one-to-one",
            "--replace-confidence",
        ]
    )

    assert status == 0
    output = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert output["candidate_temporal_assignment_mode"].eq("one-to-one").all()
    assert "raw_confidence" in output.columns
    assert summary["assignment_mode"] == "one-to-one"
    assert summary["assignment_summary"]["bidirectional_supported_count"] == 1
