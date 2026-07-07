from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_assignment_audit import (
    build_mixture_assignment_oracle_audit,
    main as assignment_audit_main,
)


def _assignments() -> pd.DataFrame:
    rows = []
    for time_s in range(3):
        rows.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "candidate_rank": 1,
                    "source": "livox_avia",
                    "track_id": f"dominant-bad-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "mixture_final_weight": 0.9,
                    "mixture_dominant": True,
                    "state_x_m": float(time_s + 9),
                    "state_y_m": 0.0,
                    "state_z_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "candidate_rank": 2,
                    "source": "lidar_360",
                    "track_id": f"oracle-good-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "mixture_final_weight": 0.1,
                    "mixture_dominant": False,
                    "state_x_m": float(time_s + 9),
                    "state_y_m": 0.0,
                    "state_z_m": 1.0,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def test_assignment_audit_finds_low_weight_oracle_candidate() -> None:
    frames, summary, by_sequence = build_mixture_assignment_oracle_audit(
        _assignments(),
        _truth(),
    )

    assert frames["oracle_branch"].tolist() == ["raw", "raw", "raw"]
    assert frames["dominant_branch"].tolist() == ["translated", "translated", "translated"]
    assert frames["oracle_weight_rank"].tolist() == [2, 2, 2]
    assert frames["oracle_mixture_weight"].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert frames["dominant_minus_oracle_error_3d_m"].tolist() == pytest.approx(
        [10.0, 10.0, 10.0],
    )
    assert summary.loc[0, "oracle_dominant_fraction"] == 0.0
    assert summary.loc[0, "oracle_candidate_error_3d_m_mse"] == 0.0
    assert summary.loc[0, "dominant_candidate_error_3d_m_mse"] == 100.0
    assert by_sequence.loc[0, "sequence_id"] == "seqA"
    assert by_sequence.loc[0, "state_error_3d_m_mse"] == 81.0


def test_assignment_audit_cli_writes_artifacts(tmp_path: Path) -> None:
    assignments_csv = tmp_path / "assignments.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _assignments().to_csv(assignments_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = assignment_audit_main(
        [
            "--assignments-csv",
            str(assignments_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_mixture_assignment_oracle_frames.csv").exists()
    summary_csv = output_dir / "mmuad_mixture_assignment_oracle_summary.csv"
    by_sequence_csv = output_dir / "mmuad_mixture_assignment_oracle_by_sequence.csv"
    summary_json = output_dir / "mmuad_mixture_assignment_oracle_summary.json"
    assert summary_csv.exists()
    assert by_sequence_csv.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["summary"][0]["oracle_dominant_fraction"] == 0.0
    assert payload["by_sequence"][0]["sequence_id"] == "seqA"
