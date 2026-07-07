from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir_mixture_gap_frames import (
    build_frame_gap_table,
    main as gap_main,
    summarize_frame_gap,
)


def _estimates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB"],
            "time_s": [0.0, 1.0, 0.0],
            "position_error_3d_m": [1.0, 3.0, 2.0],
            "state_x_m": [0.0, 1.0, 5.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [1.0, 1.0, 1.0],
            "mixture_effective_candidate_count": [1.2, 2.0, 1.0],
        }
    )


def _oracle_frames() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqB"],
            "time_s": [0.0, 1.0, 0.0],
            "candidate_count": [2, 2, 3],
            "oracle_all_3d_m": [0.0, 1.0, 2.0],
            "oracle_top3_3d_m": [0.5, 1.5, 2.0],
        }
    )


def test_frame_gap_table_computes_assignment_gap_columns() -> None:
    gap = build_frame_gap_table(_estimates(), _oracle_frames())

    assert len(gap) == 3
    assert "gap_to_oracle_all_3d_m" in gap.columns
    assert "gap_to_oracle_all_mse_contribution_m2" in gap.columns
    seq_a_time_1 = gap.loc[(gap["sequence_id"] == "seqA") & (gap["time_s"] == 1.0)].iloc[0]
    assert seq_a_time_1["gap_to_oracle_all_3d_m"] == pytest.approx(2.0)
    assert seq_a_time_1["gap_to_oracle_all_mse_contribution_m2"] == pytest.approx(8.0)


def test_frame_gap_summary_reports_pooled_and_by_sequence_mse() -> None:
    gap = build_frame_gap_table(_estimates(), _oracle_frames())
    pooled = summarize_frame_gap(gap)
    by_sequence = summarize_frame_gap(gap, group_column="sequence_id")

    assert pooled.loc[0, "mixture_mse_3d_m2"] == pytest.approx((1.0 + 9.0 + 4.0) / 3.0)
    assert pooled.loc[0, "oracle_all_3d_m_mse_3d_m2"] == pytest.approx((0.0 + 1.0 + 4.0) / 3.0)
    assert "frames_gap_to_oracle_all_3d_m_gt_1m" in pooled.columns
    seq_a = by_sequence.loc[by_sequence["sequence_id"] == "seqA"].iloc[0]
    assert seq_a["frame_count"] == 2
    assert seq_a["gap_to_oracle_all_3d_m_mean_3d_m"] == pytest.approx(1.5)


def test_gap_cli_writes_artifacts(tmp_path: Path) -> None:
    estimates_csv = tmp_path / "estimates.csv"
    oracle_csv = tmp_path / "oracle.csv"
    frame_csv = tmp_path / "gap_frames.csv"
    summary_csv = tmp_path / "gap_summary.csv"
    by_sequence_csv = tmp_path / "gap_by_sequence.csv"
    summary_json = tmp_path / "gap_summary.json"
    _estimates().to_csv(estimates_csv, index=False)
    _oracle_frames().to_csv(oracle_csv, index=False)

    status = gap_main(
        [
            "--estimates-csv",
            str(estimates_csv),
            "--oracle-frame-csv",
            str(oracle_csv),
            "--output-frame-csv",
            str(frame_csv),
            "--output-summary-csv",
            str(summary_csv),
            "--output-by-sequence-csv",
            str(by_sequence_csv),
            "--output-json",
            str(summary_json),
        ]
    )

    assert status == 0
    assert frame_csv.exists()
    assert summary_csv.exists()
    assert by_sequence_csv.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 3
    assert payload["pooled"][0]["mixture_mse_3d_m2"] == pytest.approx((1.0 + 9.0 + 4.0) / 3.0)


def test_gap_frame_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-reservoir-mixture-gap-frames"]
        == "raft_uav.mmuad.candidate_reservoir_mixture_gap_frames:main"
    )
