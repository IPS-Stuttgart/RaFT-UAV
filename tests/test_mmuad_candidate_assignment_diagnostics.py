from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_assignment_diagnostics import (
    CandidateAssignmentDiagnosticsConfig,
    build_candidate_assignment_diagnostics,
    main as assignment_diagnostics_main,
)


def _assignment_rows() -> pd.DataFrame:
    records = []
    # Frame 0: a good raw candidate is buried behind a high-weight translated candidate.
    records.extend(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "candidate_rank": 1,
                "source": "livox",
                "track_id": "bad0",
                "candidate_branch": "translated",
                "x_m": 10.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "mixture_final_weight": 0.9,
                "mixture_dominant": True,
                "state_x_m": 9.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "candidate_rank": 2,
                "source": "lidar",
                "track_id": "good0",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "mixture_final_weight": 0.1,
                "mixture_dominant": False,
                "state_x_m": 9.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
            },
        ]
    )
    # Frame 1: no candidate is near truth.
    records.extend(
        [
            {
                "sequence_id": "seqA",
                "time_s": 1.0,
                "candidate_rank": 1,
                "source": "lidar",
                "track_id": "bad1",
                "candidate_branch": "raw",
                "x_m": 12.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "mixture_final_weight": 1.0,
                "mixture_dominant": True,
                "state_x_m": 12.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
            }
        ]
    )
    # Frame 2: good dominant assignment is covered.
    records.extend(
        [
            {
                "sequence_id": "seqA",
                "time_s": 2.0,
                "candidate_rank": 1,
                "source": "lidar",
                "track_id": "good2",
                "candidate_branch": "raw",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "mixture_final_weight": 1.0,
                "mixture_dominant": True,
                "state_x_m": 2.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
            }
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
            "z_m": [0.0, 0.0, 0.0],
        }
    )


def test_assignment_diagnostics_classify_buried_missing_and_covered() -> None:
    frames, summary = build_candidate_assignment_diagnostics(
        _assignment_rows(),
        _truth_rows(),
        config=CandidateAssignmentDiagnosticsConfig(
            good_candidate_threshold_m=5.0,
            regret_threshold_m=2.0,
            top_k=1,
        ),
    )

    modes = dict(zip(frames["time_s"], frames["assignment_failure_mode"], strict=True))
    assert modes[0.0] == "good_candidate_buried"
    assert modes[1.0] == "missing_good_candidate_in_assignments"
    assert modes[2.0] == "covered"
    first = frames.loc[frames["time_s"] == 0.0].iloc[0]
    assert first["oracle_candidate_branch"] == "raw"
    assert first["dominant_candidate_branch"] == "translated"
    assert first["state_regret_m"] > 8.0
    pooled = summary.loc[
        (summary["sequence_id"] == "__pooled__")
        & (summary["assignment_failure_mode"] == "__all__")
    ].iloc[0]
    assert pooled["frame_count"] == 3
    assert pooled["oracle_in_topk_by_weight_rate"] < 1.0


def test_assignment_diagnostics_cli_writes_artifacts(tmp_path: Path) -> None:
    assignments_csv = tmp_path / "assignments.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _assignment_rows().to_csv(assignments_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = assignment_diagnostics_main(
        [
            "--assignments-csv",
            str(assignments_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
        ]
    )

    assert status == 0
    frame_csv = output_dir / "mmuad_candidate_assignment_diagnostics.csv"
    summary_csv = output_dir / "mmuad_candidate_assignment_summary.csv"
    summary_json = output_dir / "mmuad_candidate_assignment_summary.json"
    assert frame_csv.exists()
    assert summary_csv.exists()
    assert summary_json.exists()
    frame_rows = pd.read_csv(frame_csv)
    assert "state_regret_m" in frame_rows.columns
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 3
    assert payload["config"]["top_k"] == 1
