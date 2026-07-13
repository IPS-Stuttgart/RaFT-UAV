from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pandas as pd

from raft_uav.mmuad.track5_submission_ensemble import parse_submission_input
from raft_uav.mmuad.track5_submission_ensemble_grid import (
    evaluate_submission_ensemble_weight_grid,
    generate_simplex_weight_grid,
    main as ensemble_grid_main,
    write_submission_ensemble_weight_grid_outputs,
)


def _submission_rows(offset: float = 0.0, classification: int = 1) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0002"],
            "Timestamp": [0.0, 1.0, 0.0],
            "Position": [
                f"({0.0 + offset}, 0.0, 1.0)",
                f"({2.0 + offset}, 0.0, 1.0)",
                f"({10.0 + offset}, 1.0, 2.0)",
            ],
            "Classification": [classification, classification, 2],
        }
    )


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0002"],
            "time_s": [0.0, 1.0, 0.0],
            "x_m": [0.0, 2.0, 10.0],
            "y_m": [0.0, 0.0, 1.0],
            "z_m": [1.0, 1.0, 2.0],
            "class_id": [1, 1, 2],
        }
    )


def test_generate_simplex_weight_grid_can_exclude_singletons() -> None:
    weights = generate_simplex_weight_grid(2, step=0.5, include_singletons=False)

    assert weights == [(0.5, 0.5)]


def test_submission_ensemble_weight_grid_selects_best_submission(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    _submission_rows(offset=0.0, classification=1).to_csv(good, index=False)
    _submission_rows(offset=10.0, classification=3).to_csv(bad, index=False)

    summary, by_sequence, best_weights, best_policy = evaluate_submission_ensemble_weight_grid(
        [parse_submission_input(f"good={good}"), parse_submission_input(f"bad={bad}")],
        truth=_truth_rows(),
        weight_grid=generate_simplex_weight_grid(2, step=0.5),
        class_policies=("weighted-vote",),
    )

    assert tuple(best_weights) == (1.0, 0.0)
    assert best_policy == "weighted-vote"
    assert summary.iloc[0]["pose_mse"] == 0.0
    assert summary.iloc[0]["class_accuracy"] == 1.0
    assert set(by_sequence["sequence_id"]) == {"seq0001", "seq0002"}


def test_submission_ensemble_grid_writes_best_upload_artifacts(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    template = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _submission_rows(offset=0.0, classification=1).to_csv(good, index=False)
    _submission_rows(offset=10.0, classification=3).to_csv(bad, index=False)
    _submission_rows(offset=0.0, classification=1).to_csv(template, index=False)

    paths = write_submission_ensemble_weight_grid_outputs(
        submission_inputs=[parse_submission_input(f"good={good}"), parse_submission_input(f"bad={bad}")],
        truth=_truth_rows(),
        weight_grid=(weights for weights in generate_simplex_weight_grid(2, step=0.5)),
        output_dir=output_dir,
        template=pd.read_csv(template),
        class_policies=(policy for policy in ("weighted-vote", "first")),
    )

    assert paths["summary_csv"].exists()
    assert paths["best_zip"].exists()
    assert paths["best_validation_json"].exists()
    summary = pd.read_csv(paths["summary_csv"])
    assert len(summary) == 6
    assert set(summary["class_policy"]) == {"weighted-vote", "first"}
    manifest = json.loads(paths["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["class_policies"] == ["weighted-vote", "first"]
    assert manifest["grid_row_count"] == 6
    assert manifest["best_weights"] == [1.0, 0.0]
    assert manifest["best"]["pose_mse"] == 0.0


def test_submission_ensemble_grid_cli_and_entrypoint(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    bad = tmp_path / "bad.csv"
    truth = tmp_path / "truth.csv"
    template = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _submission_rows(offset=0.0, classification=1).to_csv(good, index=False)
    _submission_rows(offset=10.0, classification=3).to_csv(bad, index=False)
    _truth_rows().to_csv(truth, index=False)
    _submission_rows(offset=0.0, classification=1).to_csv(template, index=False)

    status = ensemble_grid_main(
        [
            "--submission",
            f"good={good}",
            "--submission",
            f"bad={bad}",
            "--truth-csv",
            str(truth),
            "--template",
            str(template),
            "--output-dir",
            str(output_dir),
            "--weight-step",
            "0.5",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_track5_submission_ensemble_weight_grid.csv").exists()
    assert (output_dir / "best_submission_ensemble" / "ug2_submission_ensemble.zip").exists()
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-submission-ensemble-grid"]
        == "raft_uav.mmuad.track5_submission_ensemble_grid:main"
    )
