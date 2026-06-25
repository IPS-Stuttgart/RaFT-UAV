from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    main,
    summarize_candidate_reservoir,
)


def test_candidate_reservoir_preserves_branch_candidates_beyond_global_top() -> None:
    raw = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [1.0, 1.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["raw-a", "raw-b"],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 1.0],
            "z_m": [0.0, 1.0],
            "ranker_score": [0.10, 0.20],
        }
    )
    translated = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [1.0, 1.0],
            "source": ["lidar_360", "livox_avia"],
            "track_id": ["translated-a", "translated-b"],
            "x_m": [10.0, 11.0],
            "y_m": [10.0, 11.0],
            "z_m": [10.0, 11.0],
            "ranker_score": [0.90, 0.80],
        }
    )

    reservoir = build_candidate_reservoir(
        {"raw": raw, "translated": translated},
        config=ReservoirConfig(per_source_top_n=0, per_branch_top_n=1, global_top_n=1),
    )

    assert set(reservoir["candidate_branch"]) == {"raw", "translated"}
    assert len(reservoir) == 2
    selected_by = dict(zip(reservoir["track_id"].astype(str), reservoir["reservoir_selected_by"]))
    assert "per_branch_top1:raw" in selected_by["raw-b"]
    assert "global_top1" in selected_by["translated-a"]


def test_candidate_reservoir_summary_reports_branches_and_frame_counts() -> None:
    candidates = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [1.0, 1.0, 2.0],
            "source": ["lidar_360", "lidar_360", "lidar_360"],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 1.0, 2.0],
            "z_m": [0.0, 1.0, 2.0],
            "confidence": [0.5, 0.7, 0.9],
        }
    )
    config = ReservoirConfig(per_source_top_n=1, per_branch_top_n=0, global_top_n=0)
    reservoir = build_candidate_reservoir({"static": candidates}, config=config)
    summary = summarize_candidate_reservoir(reservoir, config=config)

    assert summary["rows"] == 2
    assert summary["frames"] == 2
    assert summary["branches"] == ["static"]
    assert summary["rows_by_branch"] == {"static": 2}


def test_candidate_reservoir_cli_writes_csv_and_summary(tmp_path: Path) -> None:
    raw_csv = tmp_path / "raw.csv"
    translated_csv = tmp_path / "translated.csv"
    out_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [1.0],
            "source": ["lidar_360"],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
            "ranker_score": [0.1],
        }
    ).to_csv(raw_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seq0001"],
            "time_s": [1.0],
            "source": ["lidar_360"],
            "x_m": [10.0],
            "y_m": [10.0],
            "z_m": [10.0],
            "ranker_score": [0.9],
        }
    ).to_csv(translated_csv, index=False)

    rc = main(
        [
            "--branch",
            f"raw={raw_csv}",
            "--branch",
            f"translated={translated_csv}",
            "--output-csv",
            str(out_csv),
            "--summary-json",
            str(summary_json),
            "--per-branch-top-n",
            "1",
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "0",
        ]
    )

    assert rc == 0
    rows = pd.read_csv(out_csv)
    assert set(rows["candidate_branch"]) == {"raw", "translated"}
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["rows"] == 2
    assert summary["frames"] == 1
