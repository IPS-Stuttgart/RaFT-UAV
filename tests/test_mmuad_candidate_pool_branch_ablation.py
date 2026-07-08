from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from raft_uav.mmuad.candidate_pool_branch_ablation import (
    build_candidate_pool_branch_ablation_pools,
    build_candidate_pool_branch_ablation_tables,
    main as ablation_main,
)


def _candidate_rows() -> pd.DataFrame:
    records = []
    for time_s in range(3):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"raw-good-{time_s}",
                    "candidate_branch": "raw",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.1,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "source_translation",
                    "track_id": f"translated-bad-{time_s}",
                    "candidate_branch": "translated",
                    "x_m": float(time_s + 10.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.9,
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


def test_branch_ablation_pool_builder_creates_full_without_and_only() -> None:
    pools, manifest = build_candidate_pool_branch_ablation_pools(
        _candidate_rows(),
        group_column="candidate_branch",
    )

    assert set(pools) == {
        "full_pool",
        "without_candidate_branch_raw",
        "without_candidate_branch_translated",
        "only_candidate_branch_raw",
        "only_candidate_branch_translated",
    }
    assert len(pools["without_candidate_branch_raw"]) == 3
    assert len(pools["only_candidate_branch_raw"]) == 3
    assert set(manifest["ablation_type"]) == {"full_pool", "without_group", "only_group"}


def test_branch_ablation_tables_expose_harmful_branch_removal() -> None:
    _, pooled, _, _, manifest = build_candidate_pool_branch_ablation_tables(
        _candidate_rows(),
        _truth_rows(),
        group_column="candidate_branch",
        top_k_values=(1,),
        score_column="ranker_score",
    )

    assert not manifest.empty
    full = pooled.loc[pooled["pool_label"] == "full_pool"].iloc[0]
    without_raw = pooled.loc[pooled["pool_label"] == "without_candidate_branch_raw"].iloc[0]
    only_raw = pooled.loc[pooled["pool_label"] == "only_candidate_branch_raw"].iloc[0]
    assert full["oracle_all_mse_delta"] == 0.0
    assert only_raw["oracle_all_mse_delta"] == 0.0
    assert without_raw["oracle_all_mse_delta"] > 0.0
    assert without_raw["group_value"] == "raw"


def test_branch_ablation_cli_writes_outputs(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidate_rows().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    status = ablation_main(
        [
            "--candidate",
            f"mixed={candidates_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
            "--score-column",
            "ranker_score",
        ]
    )

    assert status == 0
    pooled_csv = output_dir / "mmuad_candidate_pool_compare_pooled.csv"
    manifest_csv = output_dir / "mmuad_candidate_pool_branch_ablation_manifest.csv"
    summary_json = output_dir / "mmuad_candidate_pool_branch_ablation_summary.json"
    assert pooled_csv.exists()
    assert manifest_csv.exists()
    assert summary_json.exists()
    pooled = pd.read_csv(pooled_csv)
    assert "ablation_type" in pooled.columns
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["pool_count"] == 5
    assert payload["worst_by_oracle_all_mse_delta"]["pool_label"] == (
        "without_candidate_branch_raw"
    )
