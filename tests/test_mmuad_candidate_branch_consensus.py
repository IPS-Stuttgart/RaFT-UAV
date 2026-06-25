from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_branch_consensus import (
    attach_candidate_branch_consensus,
    branch_consensus_summary,
    main as branch_consensus_main,
)
from raft_uav.mmuad.schema import CandidateFrame


def _branch_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq001", "seq001", "seq001", "seq001"],
            "time_s": [0.0, 0.0, 0.0, 0.0],
            "source": ["lidar_360", "lidar_360", "livox_avia", "radar_enhance_pcl"],
            "track_id": ["cluster@raw", "cluster@calibrated", "livox", "radar"],
            "candidate_branch": ["raw", "source_translation", "raw", "raw"],
            "mmuad_calibration_origin_row": [7, 7, 12, 13],
            "x_m": [10.0, 1.0, 1.2, 1.4],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0, 0.0],
            "ranker_score": [0.9, 0.4, 0.5, 0.3],
            "confidence": [0.9, 0.4, 0.5, 0.3],
        }
    )


def test_consensus_prefers_calibrated_branch_with_cross_sensor_support() -> None:
    augmented = attach_candidate_branch_consensus(
        CandidateFrame(_branch_candidates()),
        time_window_s=0.01,
        distance_gate_m=2.0,
        distance_scale_m=2.0,
        consensus_weight=2.0,
        pair_advantage_weight=1.0,
    ).rows.set_index("track_id")

    raw = augmented.loc["cluster@raw"]
    calibrated = augmented.loc["cluster@calibrated"]

    assert raw["branch_consensus_nearest_cross_source_distance_m"] == pytest.approx(8.6)
    assert calibrated["branch_consensus_nearest_cross_source_distance_m"] == pytest.approx(0.2)
    assert calibrated["branch_consensus_neighbor_count"] == 2
    assert calibrated["branch_consensus_pair_advantage_m"] > 0.0
    assert raw["branch_consensus_pair_advantage_m"] < 0.0
    assert calibrated["branch_consensus_rank_score"] > raw["branch_consensus_rank_score"]


def test_same_source_raw_calibrated_siblings_do_not_support_each_other() -> None:
    rows = _branch_candidates().iloc[:2].copy()
    augmented = attach_candidate_branch_consensus(
        CandidateFrame(rows),
        time_window_s=0.1,
        distance_gate_m=20.0,
    ).rows

    assert augmented["branch_consensus_neighbor_count"].tolist() == [0, 0]
    assert augmented["branch_consensus_nearest_cross_source_distance_m"].isna().all()


def test_branch_consensus_can_replace_confidence() -> None:
    augmented = attach_candidate_branch_consensus(
        CandidateFrame(_branch_candidates()),
        replace_confidence=True,
    ).rows

    assert augmented["confidence"].tolist() == pytest.approx(
        augmented["branch_consensus_rank_score"].tolist()
    )
    summary = branch_consensus_summary(augmented)
    assert summary["cross_source_match_count"] == 4
    assert summary["paired_hypothesis_count"] == 2


def test_branch_consensus_cli_writes_candidates_and_provenance(tmp_path: Path) -> None:
    input_csv = tmp_path / "branch_candidates.csv"
    output_csv = tmp_path / "branch_candidates_consensus.csv"
    provenance_json = tmp_path / "branch_consensus.json"
    _branch_candidates().to_csv(input_csv, index=False)

    status = branch_consensus_main(
        [
            "--candidate-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--provenance-json",
            str(provenance_json),
            "--distance-gate-m",
            "2",
            "--distance-scale-m",
            "2",
        ]
    )

    assert status == 0
    assert output_csv.exists()
    assert provenance_json.exists()
    output = pd.read_csv(output_csv)
    assert "branch_consensus_rank_score" in output.columns
    assert "branch_consensus_pair_advantage_m" in output.columns
    provenance = json.loads(provenance_json.read_text(encoding="utf-8"))
    assert provenance["row_count"] == 4
    assert provenance["candidate_branch_counts"] == {
        "raw": 3,
        "source_translation": 1,
    }
