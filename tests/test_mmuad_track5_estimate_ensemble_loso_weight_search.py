from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_ensemble import EstimateInput
from raft_uav.mmuad.track5_estimate_ensemble_loso_weight_search import main as loso_main
from raft_uav.mmuad.track5_estimate_ensemble_loso_weight_search import (
    run_track5_estimate_ensemble_loso_weight_search,
)
from raft_uav.mmuad.track5_estimate_ensemble_loso_weight_search import (
    write_loso_weight_search_outputs,
)


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0, 1.0],
            "Position": ["(0,0,0)"] * 4,
            "Classification": [2, 2, 1, 1],
        }
    )


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002", "seq0002"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "x_m": [0.0, 1.0, 4.0, 5.0],
            "y_m": [0.0, 0.0, 4.0, 4.0],
            "z_m": [0.0, 0.0, 4.0, 4.0],
        }
    )


def _good_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002", "seq0002"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [0.0, 1.0, 4.0, 5.0],
            "state_y_m": [0.0, 0.0, 4.0, 4.0],
            "state_z_m": [0.0, 0.0, 4.0, 4.0],
        }
    )


def _bad_estimate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002", "seq0002"],
            "time_s": [0.0, 1.0, 0.0, 1.0],
            "state_x_m": [10.0, 11.0, 14.0, 15.0],
            "state_y_m": [10.0, 10.0, 14.0, 14.0],
            "state_z_m": [10.0, 10.0, 14.0, 14.0],
        }
    )


def test_loso_weight_search_selects_weight_without_held_out_sequence(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)

    folds, predictions, summary, full_weights = run_track5_estimate_ensemble_loso_weight_search(
        [EstimateInput("good", good_csv), EstimateInput("bad", bad_csv)],
        template=_template(),
        truth=_truth(),
        weight_step=0.5,
    )

    assert set(folds["held_out_sequence"]) == {"seq0001", "seq0002"}
    assert folds["weight_good"].tolist() == [1.0, 1.0]
    assert folds["weight_bad"].tolist() == [0.0, 0.0]
    assert summary["loso_metrics"]["pose_mse_m2"] == pytest.approx(0.0)
    assert full_weights["weights"] == {"good": 1.0, "bad": 0.0}
    assert len(predictions) == 4


def test_loso_weight_search_writes_outputs_and_best_submission(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)

    paths = write_loso_weight_search_outputs(
        estimate_inputs=[EstimateInput("good", good_csv), EstimateInput("bad", bad_csv)],
        template=_template(),
        truth=_truth(),
        output_dir=tmp_path / "out",
        weight_step=0.5,
        write_full_train_submission=True,
        class_map={"seq0001": "2", "seq0002": "1"},
    )

    assert paths["fold_summary_csv"].exists()
    assert paths["predictions_csv"].exists()
    assert paths["summary_json"].exists()
    assert paths["full_weights_json"].exists()
    assert paths["full_train_official_zip"].exists()
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summary["loso_metrics"]["pose_mse_m2"] == pytest.approx(0.0)


def test_loso_weight_search_cli_writes_summary(tmp_path: Path) -> None:
    good_csv = tmp_path / "good.csv"
    bad_csv = tmp_path / "bad.csv"
    template_csv = tmp_path / "template.csv"
    truth_csv = tmp_path / "truth.csv"
    output_dir = tmp_path / "out"
    _good_estimate().to_csv(good_csv, index=False)
    _bad_estimate().to_csv(bad_csv, index=False)
    _template().to_csv(template_csv, index=False)
    _truth().to_csv(truth_csv, index=False)

    status = loso_main(
        [
            "--estimate-csv",
            f"good={good_csv}",
            "--estimate-csv",
            f"bad={bad_csv}",
            "--template",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--weight-step",
            "0.5",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_ensemble_loso_fold_summary.csv").exists()
    assert (output_dir / "mmuad_track5_ensemble_loso_summary.json").exists()


def test_loso_weight_search_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-estimate-ensemble-loso"]
        == "raft_uav.mmuad.track5_estimate_ensemble_loso_weight_search:main"
    )
