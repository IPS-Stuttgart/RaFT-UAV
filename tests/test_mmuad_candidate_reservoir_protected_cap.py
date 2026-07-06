from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    build_reservoir_summary,
    main as reservoir_main,
)


def _tight_cap_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 0.0],
            "source": ["lidar_360", "radar", "lidar_360"],
            "track_id": ["translated-top", "translated-mid", "raw-low"],
            "candidate_branch": ["translated", "translated", "raw"],
            "x_m": [10.0, 11.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [0.0, 0.0, 0.0],
            "confidence": [1.0, 0.9, 0.1],
        }
    )


def test_default_cap_protects_branch_quota_candidates() -> None:
    reservoir = build_candidate_reservoir(
        _tight_cap_rows(),
        config=ReservoirConfig(
            global_top_n=2,
            per_source_top_n=0,
            per_branch_top_n=1,
            max_candidates_per_frame=2,
            score_column="ranker_score",
            fallback_score_column="confidence",
            cap_reason_bonus=0.0,
        ),
    )

    assert set(reservoir["track_id"]) == {"translated-top", "raw-low"}
    raw = reservoir.loc[reservoir["track_id"] == "raw-low"].iloc[0]
    assert bool(raw["candidate_reservoir_protected"])
    assert "branch:raw" in raw["candidate_reservoir_reason"]
    summary = build_reservoir_summary(_tight_cap_rows(), reservoir)
    assert summary["reservoir_protected_count"] == 2


def test_can_disable_protected_reason_prefixes_for_score_only_cap() -> None:
    reservoir = build_candidate_reservoir(
        _tight_cap_rows(),
        config=ReservoirConfig(
            global_top_n=2,
            per_source_top_n=0,
            per_branch_top_n=1,
            max_candidates_per_frame=2,
            fallback_score_column="confidence",
            preserve_reason_prefixes=(),
        ),
    )

    assert set(reservoir["track_id"]) == {"translated-top", "translated-mid"}
    assert not reservoir["candidate_reservoir_protected"].any()


def test_candidate_reservoir_cli_can_disable_protected_cap_reasons(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    output_csv = tmp_path / "reservoir.csv"
    summary_json = tmp_path / "summary.json"
    _tight_cap_rows().to_csv(candidates_csv, index=False)

    status = reservoir_main(
        [
            "--candidate-csv",
            f"candidates={candidates_csv}",
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--global-top-n",
            "2",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "1",
            "--max-candidates-per-frame",
            "2",
            "--disable-preserved-reason-prefixes",
        ]
    )

    assert status == 0
    reservoir = pd.read_csv(output_csv)
    assert set(reservoir["track_id"]) == {"translated-top", "translated-mid"}
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["reservoir_protected_count"] == 0
