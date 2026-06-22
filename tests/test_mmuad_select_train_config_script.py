from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_select_train_config.py"
spec = importlib.util.spec_from_file_location("mmuad_select_train_config", MODULE_PATH)
assert spec is not None and spec.loader is not None
selector = importlib.util.module_from_spec(spec)
sys.modules["mmuad_select_train_config"] = selector
spec.loader.exec_module(selector)


def test_select_train_config_dry_run_writes_plan(tmp_path: Path) -> None:
    out = tmp_path / "out"

    rc = selector.main(
        [
            "--train-root",
            str(tmp_path / "train"),
            "--train-truth",
            str(tmp_path / "train_truth.csv"),
            "--train-reference",
            str(tmp_path / "train_reference.csv"),
            "--output-dir",
            str(out),
            "--source-alpha-grid",
            "0,0.5,1",
            "--ranker-spec",
            "sklearn-logistic:good_cluster_5m",
            "--dry-run",
        ]
    )

    assert rc == 0
    summary = json.loads((out / "mmuad_train_config_selector_summary.json").read_text())
    assert summary["status"] == "dry_run"
    assert summary["plan"]["source_alpha_grid"] == [0.0, 0.5, 1.0]
    assert summary["plan"]["ranker_specs"] == [
        {"model_type": "sklearn-logistic", "target_column": "good_cluster_5m"}
    ]


def test_source_alpha_loso_selects_translation_for_constant_offset() -> None:
    candidates = CandidateFrame(
        normalize_candidate_columns(
            pd.DataFrame(
                [
                    {"sequence_id": "seq1", "time_s": 0.0, "source": "lidar", "x_m": 10, "y_m": 0, "z_m": 0},
                    {"sequence_id": "seq2", "time_s": 0.0, "source": "lidar", "x_m": 10, "y_m": 0, "z_m": 0},
                ]
            )
        )
    )
    truth = pd.DataFrame(
        [
            {"sequence_id": "seq1", "time_s": 0.0, "x_m": 0, "y_m": 0, "z_m": 0},
            {"sequence_id": "seq2", "time_s": 0.0, "x_m": 0, "y_m": 0, "z_m": 0},
        ]
    )
    args = argparse.Namespace(
        source_alpha_grid="0,1",
        max_truth_time_delta_s=0.5,
        max_pair_distance_m=120.0,
    )

    rows = selector.select_source_alpha_loso(candidates, truth, args)

    best = rows.iloc[0]
    assert best["source_calibration_mode"] == "source-translation"
    assert best["source_translation_alpha"] == 1.0
    assert best["train_loso_pose_mse_loss_m2"] == 0.0
