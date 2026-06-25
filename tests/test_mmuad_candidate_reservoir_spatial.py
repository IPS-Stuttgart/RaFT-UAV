from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_spatial import (
    main as spatial_main,
    spatial_diversity_cap_reservoir,
    spatial_diversity_summary,
)


def _dense_wrong_cluster_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [0.0] * 4,
            "source": ["lidar_360"] * 4,
            "candidate_branch": ["static"] * 4,
            "track_id": ["wrong1", "wrong2", "wrong3", "good_far"],
            "x_m": [10.0, 10.1, 10.2, 0.0],
            "y_m": [0.0] * 4,
            "z_m": [1.0] * 4,
            "candidate_reservoir_score": [1.0, 0.99, 0.98, 0.80],
            "confidence": [1.0, 0.99, 0.98, 0.80],
        }
    )


def _branch_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 4,
            "time_s": [1.0] * 4,
            "source": ["lidar_360", "lidar_360", "livox_avia", "livox_avia"],
            "candidate_branch": ["translated", "translated", "raw", "raw"],
            "track_id": ["translated1", "translated2", "raw1", "raw2"],
            "x_m": [10.0, 11.0, 0.0, 1.0],
            "y_m": [0.0] * 4,
            "z_m": [1.0] * 4,
            "candidate_reservoir_score": [0.99, 0.98, 0.10, 0.09],
            "confidence": [0.99, 0.98, 0.10, 0.09],
        }
    )


def test_spatial_diversity_keeps_geometrically_distinct_candidate() -> None:
    rows = _dense_wrong_cluster_rows()
    score_only = spatial_diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=2,
        min_per_source=0,
        min_per_branch=0,
        spatial_diversity_weight=0.0,
    )
    spatial = spatial_diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=2,
        min_per_source=0,
        min_per_branch=0,
        spatial_diversity_weight=2.0,
        spatial_diversity_scale_m=5.0,
        spatial_distance_cap_m=20.0,
    )

    assert set(score_only["track_id"]) == {"wrong1", "wrong2"}
    assert set(spatial["track_id"]) == {"wrong1", "good_far"}
    good = spatial.loc[spatial["track_id"] == "good_far"].iloc[0]
    assert "spatial_fill" in good["candidate_spatial_cap_reason"]
    assert good["candidate_spatial_min_distance_m"] >= 9.0
    assert good["candidate_spatial_selection_utility"] > 1.0


def test_spatial_diversity_still_preserves_branch_and_source_quotas() -> None:
    capped = spatial_diversity_cap_reservoir(
        _branch_rows(),
        max_candidates_per_frame=3,
        min_per_source=1,
        min_per_branch=1,
        spatial_diversity_weight=1.0,
    )

    assert set(capped["candidate_branch"]) == {"translated", "raw"}
    raw = capped.loc[capped["candidate_branch"] == "raw"].iloc[0]
    assert "branch:raw" in raw["candidate_spatial_cap_reason"]
    assert "source:livox_avia" in raw["candidate_spatial_cap_reason"]


def test_spatial_diversity_summary_reports_distance_and_reasons() -> None:
    rows = _dense_wrong_cluster_rows()
    capped = spatial_diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=2,
        min_per_source=0,
        min_per_branch=0,
        spatial_diversity_weight=2.0,
        spatial_diversity_scale_m=5.0,
    )

    summary = spatial_diversity_summary(rows, capped)

    assert summary["input_rows"] == 4
    assert summary["output_rows"] == 2
    assert summary["selected_min_distance_mean_m"] >= 9.0
    assert summary["spatial_cap_reason_counts"]["score_seed"] == 1
    assert summary["spatial_cap_reason_counts"]["spatial_fill"] == 1


def test_spatial_diversity_cli_writes_outputs_and_oracle(tmp_path: Path) -> None:
    input_csv = tmp_path / "reservoir.csv"
    output_csv = tmp_path / "spatial.csv"
    summary_json = tmp_path / "summary.json"
    truth_csv = tmp_path / "truth.csv"
    oracle_summary_csv = tmp_path / "oracle_summary.csv"
    _dense_wrong_cluster_rows().to_csv(input_csv, index=False)
    pd.DataFrame(
        {
            "sequence_id": ["seqA"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [1.0],
        }
    ).to_csv(truth_csv, index=False)

    status = spatial_main(
        [
            "--input-csv",
            str(input_csv),
            "--output-csv",
            str(output_csv),
            "--summary-json",
            str(summary_json),
            "--truth-csv",
            str(truth_csv),
            "--oracle-summary-csv",
            str(oracle_summary_csv),
            "--max-candidates-per-frame",
            "2",
            "--min-per-source",
            "0",
            "--min-per-branch",
            "0",
            "--spatial-diversity-weight",
            "2",
            "--spatial-diversity-scale-m",
            "5",
            "--top-k",
            "2",
            "--max-truth-time-delta-s",
            "0.1",
        ]
    )

    assert status == 0
    output = pd.read_csv(output_csv)
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    oracle = pd.read_csv(oracle_summary_csv)
    assert set(output["track_id"]) == {"wrong1", "good_far"}
    assert summary["output_rows"] == 2
    assert oracle.loc[0, "oracle_top2_3d_m_mse"] == 0.0
