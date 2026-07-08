from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_leaderboard_pipeline import main as pipeline_main
from raft_uav.mmuad.track5_leaderboard_pipeline import run_track5_leaderboard_pipeline


def _candidates() -> pd.DataFrame:
    rows = []
    for time_s in range(5):
        rows.extend(
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
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "livox_avia",
                    "track_id": f"translated-bad-{time_s}",
                    "candidate_branch": "source_translation",
                    "x_m": float(time_s + 20.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.99,
                    "predicted_sigma_m": 30.0,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 5,
            "time_s": np.arange(5, dtype=float),
            "x_m": np.arange(5, dtype=float),
            "y_m": np.zeros(5),
            "z_m": np.ones(5),
        }
    )


def _template() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA"],
            "Timestamp": [0.0, 2.0, 4.0],
            "Position": ["(0,0,0)", "(0,0,0)", "(0,0,0)"],
            "Classification": [2, 2, 2],
        }
    )


def test_track5_leaderboard_pipeline_packages_template_ready_zip(tmp_path: Path) -> None:
    result = run_track5_leaderboard_pipeline(
        candidates=_candidates(),
        template=_template(),
        output_dir=tmp_path,
        truth=_truth(),
        class_map={"seqA": "2"},
        submission_resample_method="nearest",
        submission_max_interpolation_gap_s=1.5,
    )

    manifest = result["manifest"]
    assert manifest["schema"] == "raft-uav-mmuad-track5-leaderboard-pipeline-v3"
    assert manifest["reservoir_rows"] == 10
    assert manifest["mixture_estimate_rows"] == 5
    assert manifest["template_row_count"] == 3
    assert manifest["submission_resample_method"] == "nearest"
    assert manifest["submission_max_interpolation_gap_s"] == 1.5
    assert manifest["final_regularizer_enabled"] is False
    submission_manifest = json.loads(
        Path(result["submission_paths"]["manifest_json"]).read_text(encoding="utf-8")
    )
    assert submission_manifest["resample_method"] == "nearest"
    assert submission_manifest["max_interpolation_gap_s"] == 1.5
    validation = json.loads(
        Path(result["submission_paths"]["validation_json"]).read_text(encoding="utf-8")
    )
    assert validation["leaderboard_ready"] is True
    official = pd.read_csv(result["submission_paths"]["official_results_csv"])
    assert official["Classification"].tolist() == [2, 2, 2]
    with ZipFile(result["submission_paths"]["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_track5_leaderboard_pipeline_can_apply_final_regularizer(tmp_path: Path) -> None:
    result = run_track5_leaderboard_pipeline(
        candidates=_candidates(),
        template=_template(),
        output_dir=tmp_path,
        truth=_truth(),
        class_map={"seqA": "2"},
        submission_resample_method="nearest",
        apply_final_regularizer=True,
        regularizer_smoothness_weight=0.5,
        regularizer_huber_delta_m=10.0,
        regularizer_iterations=2,
        regularizer_observation_sigma_m=5.0,
    )

    manifest = result["manifest"]
    assert manifest["final_regularizer_enabled"] is True
    assert manifest["regularizer_smoothness_weight"] == 0.5
    assert manifest["regularizer_huber_delta_m"] == 10.0
    assert manifest["regularizer_iterations"] == 2
    assert manifest["regularizer_observation_sigma_m"] == 5.0
    assert result["submission_paths"]["regularized_estimates_csv"].exists()
    regularizer_manifest = json.loads(
        Path(result["submission_paths"]["manifest_json"]).read_text(encoding="utf-8")
    )
    assert regularizer_manifest["schema"] == "raft-uav-mmuad-track5-trajectory-regularizer-v1"
    assert regularizer_manifest["smoothness_weight"] == 0.5
    validation = json.loads(
        Path(result["submission_paths"]["validation_json"]).read_text(encoding="utf-8")
    )
    assert validation["leaderboard_ready"] is True
    with ZipFile(result["submission_paths"]["official_zip"]) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_track5_leaderboard_pipeline_cli_writes_manifest_and_zip(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _truth().to_csv(truth_csv, index=False)
    _template().to_csv(template_csv, index=False)
    pd.DataFrame({"sequence_id": ["seqA"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = pipeline_main(
        [
            "--candidate-csv",
            f"union={candidates_csv}",
            "--template",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--global-top-n",
            "1",
            "--per-branch-top-n",
            "1",
            "--per-source-top-n",
            "0",
            "--max-candidates-per-frame",
            "2",
            "--reservoir-score-column",
            "ranker_score",
            "--mixture-score-column",
            "candidate_reservoir_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--smoothness-weight",
            "100",
            "--iterations",
            "5",
            "--submission-resample-method",
            "nearest",
            "--submission-max-interpolation-gap-s",
            "2.0",
            "--final-regularizer",
            "--regularizer-smoothness-weight",
            "0.5",
            "--regularizer-iterations",
            "2",
            "--require-leaderboard-ready",
        ]
    )

    assert status == 0
    manifest_path = output_dir / "mmuad_track5_leaderboard_pipeline_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["submission_paths"]["official_zip"].endswith("ug2_submission.zip")
    assert manifest["submission_resample_method"] == "nearest"
    assert manifest["submission_max_interpolation_gap_s"] == 2.0
    assert manifest["final_regularizer_enabled"] is True
    assert (output_dir / "track5_submission" / "ug2_submission.zip").exists()
    assert (output_dir / "track5_submission" / "mmuad_track5_regularized_estimates.csv").exists()
    assert (output_dir / "reservoir_mixture" / "mmuad_candidate_mixture_estimates.csv").exists()


def test_track5_leaderboard_pipeline_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-track5-leaderboard-pipeline"]
        == "raft_uav.mmuad.track5_leaderboard_pipeline:main"
    )
