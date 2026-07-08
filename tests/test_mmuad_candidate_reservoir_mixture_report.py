from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_mixture_report import main as report_main


def test_reservoir_mixture_report_writes_actionable_gap_and_bottleneck(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    estimates = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "state_x_m": 10.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
                "position_error_3d_m": 10.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "state_x_m": 11.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
                "position_error_3d_m": 10.0,
            },
        ]
    )
    oracle_frames = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "oracle_all_3d_m": 1.0,
                "oracle_top1_3d_m": 1.0,
                "oracle_top3_3d_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "oracle_all_3d_m": 1.0,
                "oracle_top1_3d_m": 1.0,
                "oracle_top3_3d_m": 1.0,
            },
        ]
    )
    estimates.to_csv(run_dir / "mmuad_candidate_mixture_estimates.csv", index=False)
    oracle_frames.to_csv(run_dir / "mmuad_reservoir_mixture_oracle_frames.csv", index=False)

    status = report_main(
        [
            "--run-dir",
            str(run_dir),
            "--target-mse-3d-m2",
            "24.51",
        ]
    )

    assert status == 0
    frame_gap_csv = run_dir / "mmuad_reservoir_mixture_report_frame_gap.csv"
    bottleneck_csv = run_dir / "mmuad_reservoir_mixture_report_bottleneck.csv"
    bottleneck_by_sequence_csv = (
        run_dir / "mmuad_reservoir_mixture_report_bottleneck_by_sequence.csv"
    )
    report_json = run_dir / "mmuad_reservoir_mixture_report.json"
    assert frame_gap_csv.exists()
    assert bottleneck_csv.exists()
    assert bottleneck_by_sequence_csv.exists()
    assert report_json.exists()
    frame_gap = pd.read_csv(frame_gap_csv)
    assert frame_gap["gap_to_oracle_all_3d_m"].tolist() == [9.0, 9.0]
    bottleneck = pd.read_csv(bottleneck_csv)
    assert bottleneck.loc[0, "reservoir_oracle_all_mse_3d_m2"] == 1.0
    assert bottleneck.loc[0, "mixture_mse_3d_m2"] == 100.0
    assert bottleneck.loc[0, "primary_bottleneck"] == "assignment_limited"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 2
    assert payload["pooled_bottleneck"]["primary_bottleneck"] == "assignment_limited"
    assert payload["worst_sequence_bottleneck"]["sequence_id"] == "seqA"


def test_reservoir_mixture_report_nearest_time_join_tolerance(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.04,
                "position_error_3d_m": 5.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.04,
                "position_error_3d_m": 7.0,
            },
        ]
    ).to_csv(run_dir / "mmuad_candidate_mixture_estimates.csv", index=False)
    pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "oracle_all_3d_m": 1.0,
                "oracle_top3_3d_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "oracle_all_3d_m": 2.0,
                "oracle_top3_3d_m": 2.0,
            },
        ]
    ).to_csv(run_dir / "mmuad_reservoir_mixture_oracle_frames.csv", index=False)

    status = report_main(
        [
            "--run-dir",
            str(run_dir),
            "--time-join-tolerance-s",
            "0.05",
        ]
    )

    assert status == 0
    frame_gap = pd.read_csv(run_dir / "mmuad_reservoir_mixture_report_frame_gap.csv")
    assert len(frame_gap) == 2
    assert frame_gap["time_delta_s"].tolist() == [0.04, 0.04]
    report_json = run_dir / "mmuad_reservoir_mixture_report.json"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["config"]["time_join_tolerance_s"] == 0.05


def test_reservoir_mixture_report_preserves_numeric_looking_sequence_ids(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "state_x_m": 3.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
                "position_error_3d_m": 3.0,
            }
        ]
    ).to_csv(run_dir / "mmuad_candidate_mixture_estimates.csv", index=False)
    pd.DataFrame(
        [
            {
                "sequence_id": "001",
                "time_s": 0.0,
                "oracle_all_3d_m": 1.0,
                "oracle_top1_3d_m": 1.0,
                "oracle_top3_3d_m": 1.0,
            }
        ]
    ).to_csv(run_dir / "mmuad_reservoir_mixture_oracle_frames.csv", index=False)

    status = report_main(
        [
            "--run-dir",
            str(run_dir),
            "--target-mse-3d-m2",
            "1.0",
        ]
    )

    assert status == 0
    frame_gap = pd.read_csv(
        run_dir / "mmuad_reservoir_mixture_report_frame_gap.csv",
        dtype=str,
        keep_default_na=False,
    )
    assert frame_gap.loc[0, "sequence_id"] == "001"
    report_json = run_dir / "mmuad_reservoir_mixture_report.json"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["worst_sequence_bottleneck"]["sequence_id"] == "001"


def test_reservoir_mixture_report_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-reservoir-mixture-report"]
        == "raft_uav.mmuad.candidate_reservoir_mixture_report:main"
    )
