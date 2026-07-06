from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_mixture_map import main as reservoir_mixture_main


def _candidate_rows() -> pd.DataFrame:
    records = []
    for time_s in range(3):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"raw-good-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.1,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"translated-bad-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.9,
                    "predicted_sigma_m": 5.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def test_reservoir_mixture_cli_writes_oracle_diagnostics(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = reservoir_mixture_main(
        [
            "--candidate-csv",
            f"raw={candidates_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--global-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--max-candidates-per-frame",
            "2",
            "--reservoir-cap-reason-bonus",
            "0.5",
            "--oracle-top-k",
            "1",
            "--oracle-top-k",
            "2",
            "--score-weight",
            "0",
            "--sigma-log-weight",
            "0",
            "--loss",
            "squared",
            "--smoothness-weight",
            "100",
        ]
    )

    assert status == 0
    oracle_summary = output_dir / "mmuad_reservoir_mixture_oracle_summary.csv"
    oracle_by_sequence = output_dir / "mmuad_reservoir_mixture_oracle_by_sequence.csv"
    oracle_frames = output_dir / "mmuad_reservoir_mixture_oracle_frames.csv"
    gap_summary = output_dir / "mmuad_reservoir_mixture_gap_summary.csv"
    gap_by_sequence = output_dir / "mmuad_reservoir_mixture_gap_by_sequence.csv"
    combined_summary = output_dir / "mmuad_reservoir_mixture_summary.json"
    assert oracle_summary.exists()
    assert oracle_by_sequence.exists()
    assert oracle_frames.exists()
    assert gap_summary.exists()
    assert gap_by_sequence.exists()
    summary = pd.read_csv(oracle_summary)
    assert summary.loc[0, "oracle_all_3d_m_mse"] == 0.0
    assert summary.loc[0, "oracle_top2_3d_m_mse"] == 0.0
    assert "oracle_top3_3d_m_mse" not in summary.columns
    assert "oracle_top5_3d_m_mse" not in summary.columns
    assert "oracle_top10_3d_m_mse" not in summary.columns
    assert "oracle_top20_3d_m_mse" not in summary.columns
    gap = pd.read_csv(gap_summary)
    assert gap.loc[0, "reservoir_oracle_all_mse_3d_m2"] == 0.0
    assert "mixture_mse_3d_m2" in gap.columns
    assert "gap_to_oracle_all_mse_3d_m2" in gap.columns
    sequence_gap = pd.read_csv(gap_by_sequence)
    assert sequence_gap["sequence_id"].tolist() == ["seqA"]
    assert sequence_gap.loc[0, "reservoir_oracle_all_mse_3d_m2"] == 0.0
    assert "assignment_gap_mse_3d_m2" in sequence_gap.columns
    payload = json.loads(combined_summary.read_text(encoding="utf-8"))
    assert payload["reservoir_config"]["cap_reason_bonus"] == 0.5
    assert payload["mixture_config"]["loss"] == "squared"
    assert payload["reservoir_oracle"]["top_k_values"] == [1, 2]
    assert payload["reservoir_oracle"]["frame_count"] == 3
    assert payload["reservoir_oracle"]["pooled"]["oracle_all_3d_m_mse"] == 0.0
    assert payload["reservoir_mixture_gap"]["reservoir_oracle_all_mse_3d_m2"] == 0.0
    assert payload["reservoir_mixture_gap_by_sequence"]["sequence_count"] == 1
    assert (
        payload["reservoir_mixture_gap_by_sequence"]["worst_assignment_gap"]["sequence_id"]
        == "seqA"
    )
