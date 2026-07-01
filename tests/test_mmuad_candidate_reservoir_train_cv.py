from __future__ import annotations

import json

import pandas as pd

from raft_uav.mmuad.candidate_reservoir_train_cv import (
    main as train_cv_main,
    select_candidate_reservoir_offsets_by_sequence_cv,
)


def _candidate_rows() -> pd.DataFrame:
    records = []
    for sequence_id in ("seqA", "seqB", "seqC"):
        records.extend(
            [
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "lidar_360",
                    "track_id": f"{sequence_id}-raw-good",
                    "candidate_branch": "raw",
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.1,
                    "confidence": 0.1,
                },
                {
                    "sequence_id": sequence_id,
                    "time_s": 0.0,
                    "source": "livox_avia",
                    "track_id": f"{sequence_id}-translated-bad",
                    "candidate_branch": "translated",
                    "x_m": 20.0,
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.9,
                    "confidence": 0.9,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA", "seqB", "seqC"],
            "time_s": [0.0, 0.0, 0.0],
            "x_m": [0.0, 0.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
            "z_m": [1.0, 1.0, 1.0],
        }
    )


def test_train_cv_selector_writes_train_selected_raw_offset() -> None:
    config, folds, final_summary, best = select_candidate_reservoir_offsets_by_sequence_cv(
        _candidate_rows(),
        _truth_rows(),
        branch_offset_grid=["raw=0,1"],
        global_top_n=1,
        per_source_top_n=0,
        per_branch_top_n=0,
        max_candidates_per_frame=1,
        top_k_values=(1,),
        max_truth_time_delta_s=0.1,
        selection_metric="oracle_top1_3d_m_mse",
        write_best_reservoir=True,
    )

    assert config["selected_grid_label"] == "branch_raw_1"
    assert config["branch_score_offsets"] == {"raw": 1.0}
    assert len(folds) == 3
    assert folds["oracle_top1_3d_m_mse"].max() == 0.0
    assert final_summary.iloc[0]["oracle_top1_3d_m_mse"] == 0.0
    assert best is not None
    assert set(best["candidate_branch"]) == {"raw"}


def test_train_cv_cli_writes_config_and_fold_summary(tmp_path) -> None:
    candidate_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _candidate_rows().to_csv(candidate_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)

    rc = train_cv_main(
        [
            "--candidate",
            f"mixed={candidate_csv}",
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--branch-score-offset-grid",
            "raw=0,1",
            "--global-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--per-branch-top-n",
            "0",
            "--max-candidates-per-frame",
            "1",
            "--top-k",
            "1",
            "--max-truth-time-delta-s",
            "0.1",
            "--selection-metric",
            "oracle_top1_3d_m_mse",
            "--write-best-reservoir",
        ]
    )

    assert rc == 0
    config = json.loads(
        (output_dir / "mmuad_candidate_reservoir_train_selected_config.json").read_text(
            encoding="utf-8",
        ),
    )
    folds = pd.read_csv(output_dir / "mmuad_candidate_reservoir_train_cv_folds.csv")
    final_grid = pd.read_csv(output_dir / "mmuad_candidate_reservoir_train_final_grid_summary.csv")
    reservoir = pd.read_csv(output_dir / "mmuad_candidate_reservoir_train_selected.csv")
    assert config["selected_grid_label"] == "branch_raw_1"
    assert config["top_k_values"] == [1]
    assert "oracle_top3_3d_m_mse" not in final_grid.columns
    assert len(folds) == 3
    assert set(reservoir["candidate_branch"]) == {"raw"}
