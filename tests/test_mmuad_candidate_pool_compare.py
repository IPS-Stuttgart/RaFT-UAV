from __future__ import annotations

import json

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_pool_compare import (
    _by_reference_branch_summary,
    build_candidate_pool_compare_tables,
    main as pool_compare_main,
)


def _reference_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA", "seqA", "seqA"],
            "time_s": [0.0, 0.0, 1.0, 1.0],
            "source": ["lidar", "livox", "lidar", "livox"],
            "track_id": ["good-0", "bad-0", "good-1", "bad-1"],
            "candidate_branch": ["raw", "translated", "raw", "translated"],
            "x_m": [0.0, 20.0, 1.0, 18.0],
            "y_m": [0.0, 0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0, 1.0],
            "candidate_reservoir_score": [0.1, 0.9, 0.2, 0.95],
            "ranker_score": [0.1, 0.9, 0.2, 0.95],
        }
    )


def _pruned_candidates() -> pd.DataFrame:
    return _reference_candidates().loc[lambda frame: frame["candidate_branch"] == "translated"]


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqA"],
            "time_s": [0.0, 1.0],
            "x_m": [0.0, 1.0],
            "y_m": [0.0, 0.0],
            "z_m": [1.0, 1.0],
        }
    )


def test_candidate_pool_compare_reports_lost_oracle_candidates() -> None:
    frame_rows, pooled, by_sequence, by_branch = build_candidate_pool_compare_tables(
        _reference_candidates(),
        {"pruned": _pruned_candidates()},
        _truth_rows(),
        top_k_values=(1, 2),
        max_truth_time_delta_s=0.1,
        good_candidate_threshold_m=1.0,
    )

    assert len(frame_rows) == 2
    assert frame_rows["good_candidate_lost"].all()
    assert pooled.loc[0, "pool_label"] == "pruned"
    assert pooled.loc[0, "reference_oracle_all_mse"] == 0.0
    assert pooled.loc[0, "candidate_oracle_all_mse"] > 100.0
    assert pooled.loc[0, "good_candidate_lost_fraction"] == 1.0
    assert by_sequence.loc[0, "sequence_id"] == "seqA"
    assert by_branch.loc[0, "reference_candidate_branch"] == "raw"


def test_candidate_pool_compare_handles_all_missing_reference_branches() -> None:
    frame_rows = pd.DataFrame(
        {
            "pool_label": ["pruned", "pruned"],
            "sequence_id": ["seqA", "seqA"],
            "reference_oracle_all_candidate_branch": [np.nan, np.nan],
            "reference_oracle_all_3d_m": [1.0, 2.0],
            "candidate_oracle_all_3d_m": [1.5, 2.5],
            "pool_frame_present": [True, True],
            "reference_has_good_candidate": [True, True],
            "candidate_has_good_candidate": [True, True],
            "good_candidate_lost": [False, False],
            "oracle_ceiling_worse": [True, True],
        }
    )

    by_branch = _by_reference_branch_summary(frame_rows, top_k_values=(1, 2))

    assert by_branch.empty


def test_candidate_pool_compare_cli_writes_artifacts(tmp_path) -> None:
    reference_csv = tmp_path / "reference.csv"
    pruned_csv = tmp_path / "pruned.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _reference_candidates().to_csv(reference_csv, index=False)
    _pruned_candidates().to_csv(pruned_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    rc = pool_compare_main(
        [
            "--reference-candidate",
            f"full={reference_csv}",
            "--candidate",
            f"pruned={pruned_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
            "--top-k",
            "2",
            "--max-truth-time-delta-s",
            "0.1",
            "--good-candidate-threshold-m",
            "1.0",
        ]
    )

    assert rc == 0
    pooled = pd.read_csv(output_dir / "mmuad_candidate_pool_compare_pooled.csv")
    frame_rows = pd.read_csv(output_dir / "mmuad_candidate_pool_compare_frames.csv")
    summary = json.loads(
        (output_dir / "mmuad_candidate_pool_compare_summary.json").read_text(
            encoding="utf-8",
        ),
    )
    assert pooled.loc[0, "pool_label"] == "pruned"
    assert frame_rows["good_candidate_lost"].all()
    assert summary["pooled"][0]["good_candidate_lost_fraction"] == 1.0
