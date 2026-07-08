from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_oracle_targets import CandidateOracleTargetConfig
from raft_uav.mmuad.candidate_oracle_targets import build_candidate_oracle_targets
from raft_uav.mmuad.candidate_oracle_targets import main as oracle_targets_main


def _candidate_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "wrong-high-score",
                "candidate_branch": "translated",
                "x_m": 5.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.9,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "livox_avia",
                "track_id": "oracle-low-score",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.1,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "lidar_360",
                "track_id": "near-second",
                "candidate_branch": "raw",
                "x_m": 1.2,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.4,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "source": "dynamic",
                "track_id": "oracle-second",
                "candidate_branch": "dynamic",
                "x_m": 1.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 0.8,
            },
        ]
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [0.0, 0.0],
        }
    )


def test_candidate_oracle_targets_label_oracle_and_soft_weights() -> None:
    target_rows, frame_summary, summary = build_candidate_oracle_targets(
        _candidate_rows(),
        _truth_rows(),
        config=CandidateOracleTargetConfig(soft_tau_m=(2.0,), good_thresholds_m=(0.5,)),
    )

    assert len(frame_summary) == 2
    assert int(target_rows["candidate_is_oracle"].sum()) == 2
    first_frame = target_rows.loc[target_rows["time_s"] == 0.0]
    oracle = first_frame.loc[first_frame["candidate_is_oracle"]].iloc[0]
    assert oracle["track_id"] == "oracle-low-score"
    assert int(oracle["candidate_score_rank"]) == 2
    assert first_frame["soft_oracle_weight_tau_2_m"].sum() == pytest.approx(1.0)
    pooled = summary["pooled"]
    assert pooled["oracle_mse_3d_m2"] == pytest.approx(0.0)
    assert pooled["score_top1_mse_3d_m2"] > 0.0
    assert pooled["oracle_in_score_top1_fraction"] == pytest.approx(0.5)


def test_candidate_oracle_targets_cli_writes_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "targets"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = oracle_targets_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--score-column",
            "ranker_score",
            "--soft-tau-m",
            "2",
            "--good-threshold-m",
            "0.5",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_candidate_oracle_targets.csv").exists()
    assert (output_dir / "mmuad_candidate_oracle_target_frame_summary.csv").exists()
    summary_json = output_dir / "mmuad_candidate_oracle_target_summary.json"
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["candidate_rows"] == 4
    assert payload["frame_count"] == 2
    assert payload["pooled"]["oracle_mse_3d_m2"] == pytest.approx(0.0)


def test_candidate_oracle_targets_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-candidate-oracle-targets"]
        == "raft_uav.mmuad.candidate_oracle_targets:main"
    )
