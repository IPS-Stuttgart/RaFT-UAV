from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_mixture_assignment_summary import main as summary_main
from raft_uav.mmuad.candidate_mixture_assignment_summary import summarize_assignments


def _assignment_rows() -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "candidate_branch": "raw",
                "source": "lidar_360",
                "mixture_final_weight": 0.9,
                "mixture_dominant": True,
                "mixture_sigma_m": 1.0,
                "mixture_distance_to_state_m": 0.1,
                "mixture_raw_score": 0.2,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "candidate_branch": "translated",
                "source": "livox_avia",
                "mixture_final_weight": 0.1,
                "mixture_dominant": False,
                "mixture_sigma_m": 5.0,
                "mixture_distance_to_state_m": 4.0,
                "mixture_raw_score": 0.8,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "candidate_branch": "raw",
                "source": "lidar_360",
                "mixture_final_weight": 0.4,
                "mixture_dominant": False,
                "mixture_sigma_m": 2.0,
                "mixture_distance_to_state_m": 1.0,
                "mixture_raw_score": 0.3,
            },
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "candidate_branch": "dynamic",
                "source": "radar_enhance_pcl",
                "mixture_final_weight": 0.6,
                "mixture_dominant": True,
                "mixture_sigma_m": 1.5,
                "mixture_distance_to_state_m": 0.3,
                "mixture_raw_score": 0.7,
            },
        ]
    )


def test_assignment_summary_reports_branch_and_source_mass() -> None:
    pooled, by_sequence = summarize_assignments(_assignment_rows())

    branch_rows = pooled.loc[pooled["group_column"] == "candidate_branch"]
    raw = branch_rows.loc[branch_rows["group_value"] == "raw"].iloc[0]
    dynamic = branch_rows.loc[branch_rows["group_value"] == "dynamic"].iloc[0]
    source_rows = pooled.loc[pooled["group_column"] == "source"]

    assert raw["candidate_rows"] == 2
    assert raw["frame_count"] == 2
    assert raw["responsibility_sum"] == 1.3
    assert raw["dominant_count"] == 1
    assert dynamic["dominant_count"] == 1
    assert set(source_rows["group_value"]) == {"lidar_360", "livox_avia", "radar_enhance_pcl"}
    assert set(by_sequence["sequence_id"]) == {"seqA"}


def test_assignment_summary_cli_writes_artifacts(tmp_path: Path) -> None:
    assignments_csv = tmp_path / "assignments.csv"
    output_dir = tmp_path / "out"
    _assignment_rows().to_csv(assignments_csv, index=False)

    status = summary_main(
        [
            "--assignments-csv",
            str(assignments_csv),
            "--output-dir",
            str(output_dir),
            "--group-column",
            "candidate_branch",
            "--group-column",
            "source",
        ]
    )

    assert status == 0
    pooled_csv = output_dir / "mmuad_candidate_mixture_assignment_summary.csv"
    by_sequence_csv = output_dir / "mmuad_candidate_mixture_assignment_by_sequence.csv"
    summary_json = output_dir / "mmuad_candidate_mixture_assignment_summary.json"
    assert pooled_csv.exists()
    assert by_sequence_csv.exists()
    assert summary_json.exists()
    pooled = pd.read_csv(pooled_csv)
    assert {"candidate_branch", "source"}.issubset(set(pooled["group_column"]))
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["assignment_rows"] == 4
    assert payload["frame_count"] == 2
    assert payload["dominant_rows"] == 2
