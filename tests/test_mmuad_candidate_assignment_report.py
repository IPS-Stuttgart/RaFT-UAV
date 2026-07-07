from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd

from raft_uav.mmuad.candidate_assignment_report import main as report_main


def _assignment_rows() -> pd.DataFrame:
    records = []
    for time_s in [0.0, 1.0, 2.0]:
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": time_s,
                    "candidate_input_row": int(time_s * 2),
                    "candidate_rank": 1,
                    "source": "lidar_360",
                    "track_id": f"good-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": time_s,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "mixture_final_weight": 0.8,
                    "mixture_dominant": True,
                    "state_x_m": time_s,
                    "state_y_m": 0.0,
                    "state_z_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": time_s,
                    "candidate_input_row": int(time_s * 2 + 1),
                    "candidate_rank": 2,
                    "source": "dynamic",
                    "track_id": f"bad-{time_s}",
                    "candidate_branch": "dynamic",
                    "x_m": time_s + 20.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "mixture_final_weight": 0.2,
                    "mixture_dominant": False,
                    "state_x_m": time_s,
                    "state_y_m": 0.0,
                    "state_z_m": 1.0,
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


def test_candidate_assignment_report_writes_full_bundle(tmp_path: Path) -> None:
    assignments_csv = tmp_path / "assignments.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "report"
    _assignment_rows().to_csv(assignments_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = report_main(
        [
            "--assignments-csv",
            str(assignments_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--block-max-gap-s",
            "2.0",
            "--action-top-n-blocks",
            "5",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_candidate_assignment_diagnostics.csv").exists()
    assert (output_dir / "mmuad_candidate_assignment_summary.csv").exists()
    assert (output_dir / "mmuad_candidate_assignment_branch_summary.csv").exists()
    assert (output_dir / "mmuad_candidate_assignment_blocks.csv").exists()
    assert (output_dir / "mmuad_candidate_assignment_action_plan.csv").exists()
    assert (output_dir / "mmuad_candidate_assignment_action_summary.csv").exists()
    report_json = output_dir / "mmuad_candidate_assignment_report.json"
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "raft-uav-mmuad-candidate-assignment-report-v2"
    assert payload["frame_count"] == 3
    assert payload["block_count"] >= 1
    assert payload["action_count"] >= 1
    assert payload["pooled"]["state_error_3d_m_mse"] == 0.0
    assert "branch_summary_csv" in payload["paths"]
    assert "blocks_csv" in payload["paths"]
    assert "action_rows_csv" in payload["paths"]
    assert payload["top_action"]["recommended_action"]


def test_candidate_assignment_report_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-candidate-assignment-report"]
        == "raft_uav.mmuad.candidate_assignment_report:main"
    )
