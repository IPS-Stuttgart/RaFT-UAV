from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_offset_recommendations import (
    ReservoirOffsetRecommendationConfig,
    build_reservoir_offset_recommendations,
    main as offset_recommendations_main,
)


def _frame_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA", "seqB"],
            "assignment_failure_mode": [
                "good_candidate_buried",
                "wrong_dominant_assignment",
                "smoothing_assignment_gap",
                "covered",
                "missing_good_candidate_in_assignments",
            ],
            "oracle_candidate_branch": ["raw", "raw", "dynamic", "raw", "raw"],
            "dominant_candidate_branch": ["translated", "translated", "translated", "raw", "translated"],
            "oracle_source": ["lidar_360", "lidar_360", "livox_avia", "lidar_360", "lidar_360"],
            "dominant_source": ["radar", "radar", "radar", "lidar_360", "radar"],
            "state_regret_m": [10.0, 8.0, 4.0, 0.5, 100.0],
            "dominant_regret_m": [9.0, 8.0, 3.0, 0.0, 100.0],
        }
    )


def test_offset_recommendations_promote_oracle_branches_and_demote_dominant() -> None:
    recommendations = build_reservoir_offset_recommendations(
        _frame_rows(),
        config=ReservoirOffsetRecommendationConfig(max_abs_offset=1.0),
    )

    branches = recommendations.loc[recommendations["label_type"] == "branch"].set_index("label")
    assert branches.loc["raw", "recommended_offset"] > 0.0
    assert branches.loc["dynamic", "recommended_offset"] > 0.0
    assert branches.loc["translated", "recommended_offset"] < 0.0
    # The missing-good-candidate row is intentionally ignored because promoting
    # the oracle branch cannot recover a candidate that is absent from the assignments.
    assert branches.loc["raw", "promote_weight"] == 18.0
    assert branches.loc["translated", "demote_weight"] == 22.0

    sources = recommendations.loc[recommendations["label_type"] == "source"].set_index("label")
    assert sources.loc["lidar_360", "recommended_offset"] > 0.0
    assert sources.loc["livox_avia", "recommended_offset"] > 0.0
    assert sources.loc["radar", "recommended_offset"] < 0.0


def test_offset_recommendations_cli_writes_csv_json_and_flags(tmp_path: Path) -> None:
    frame_csv = tmp_path / "frames.csv"
    output_dir = tmp_path / "out"
    _frame_rows().to_csv(frame_csv, index=False)

    status = offset_recommendations_main(
        [
            "--frame-csv",
            str(frame_csv),
            "--output-dir",
            str(output_dir),
            "--max-abs-offset",
            "0.75",
        ]
    )

    assert status == 0
    csv_path = output_dir / "mmuad_candidate_reservoir_offset_recommendations.csv"
    json_path = output_dir / "mmuad_candidate_reservoir_offset_recommendations.json"
    cli_path = output_dir / "mmuad_candidate_reservoir_offset_cli.txt"
    assert csv_path.exists()
    assert json_path.exists()
    assert cli_path.exists()
    written = pd.read_csv(csv_path)
    assert not written.empty
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["config"]["max_abs_offset"] == 0.75
    assert payload["recommendation_count"] == len(written)
    cli_text = cli_path.read_text(encoding="utf-8")
    assert "--branch-score-offset-grid" in cli_text
    assert "--source-score-offset-grid" in cli_text
