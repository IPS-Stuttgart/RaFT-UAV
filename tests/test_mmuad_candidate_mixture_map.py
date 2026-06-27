from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    compute_candidate_responsibilities,
    main as mixture_main,
    run_candidate_mixture_map,
)


def _uncertainty_candidates() -> pd.DataFrame:
    records = []
    for time_s in range(5):
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"good-{time_s}",
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
                    "source": "dynamic",
                    "track_id": f"bad-{time_s}",
                    "candidate_branch": "dynamic",
                    "x_m": float(time_s + 10),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.9,
                    "predicted_sigma_m": 20.0,
                },
            ]
        )
    return pd.DataFrame.from_records(records)


def _truth_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 5,
            "time_s": np.arange(5, dtype=float),
            "x_m": np.arange(5, dtype=float),
            "y_m": np.zeros(5),
            "z_m": np.ones(5),
        }
    )


def test_branch_balance_preserves_mass_for_small_candidate_branch() -> None:
    rows = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "raw-near",
                "candidate_branch": "raw",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "raw-offset",
                "candidate_branch": "raw",
                "x_m": 0.5,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "dynamic",
                "track_id": "translated-far",
                "candidate_branch": "translated",
                "x_m": 10.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )
    responsibilities = compute_candidate_responsibilities(
        rows,
        np.asarray([0.0, 0.0, 0.0]),
        config=CandidateMixtureMapConfig(
            top_k=0,
            score_column="ranker_score",
            score_weight=0.0,
            sigma_log_weight=0.0,
            branch_balance=1.0,
        ),
    )

    mass = responsibilities.groupby("candidate_branch")["mixture_responsibility"].sum()
    assert mass["raw"] == pytest.approx(0.5)
    assert mass["translated"] == pytest.approx(0.5)


def test_candidate_mixture_uses_learned_sigma_to_reject_high_score_clutter() -> None:
    result = run_candidate_mixture_map(
        _uncertainty_candidates(),
        truth=_truth_rows(),
        config=CandidateMixtureMapConfig(
            top_k=2,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=100.0,
            iterations=5,
        ),
    )

    expected = np.arange(5, dtype=float)
    assert np.max(np.abs(result.estimates["state_x_m"].to_numpy(float) - expected)) < 0.05
    dominant = result.assignments.loc[result.assignments["mixture_dominant"]]
    assert dominant["track_id"].astype(str).str.startswith("good-").all()
    assert result.summary["metrics"]["pooled"]["rmse_3d_m"] < 0.05


def test_candidate_mixture_smoothness_recovers_from_one_high_score_outlier() -> None:
    records = []
    for time_s in range(5):
        good_score = 0.5 if time_s == 2 else 0.9
        bad_score = 0.9 if time_s == 2 else 0.1
        records.extend(
            [
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "lidar_360",
                    "track_id": f"good-{time_s}",
                    "x_m": float(time_s),
                    "y_m": 0.0,
                    "z_m": 0.0,
                    "ranker_score": good_score,
                    "predicted_sigma_m": 1.0,
                },
                {
                    "sequence_id": "seqA",
                    "time_s": float(time_s),
                    "source": "dynamic",
                    "track_id": f"bad-{time_s}",
                    "x_m": float(20 + time_s),
                    "y_m": 0.0,
                    "z_m": 0.0,
                    "ranker_score": bad_score,
                    "predicted_sigma_m": 1.0,
                },
            ]
        )

    result = run_candidate_mixture_map(
        pd.DataFrame.from_records(records),
        config=CandidateMixtureMapConfig(
            top_k=2,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=100.0,
            iterations=5,
            loss="huber",
            huber_delta=1.0,
        ),
    )

    middle = result.estimates.loc[result.estimates["time_s"] == 2.0, "state_x_m"].iloc[0]
    assert middle == pytest.approx(2.0, abs=0.1)
    middle_assignments = result.assignments.loc[result.assignments["time_s"] == 2.0]
    dominant = middle_assignments.loc[middle_assignments["mixture_dominant"]].iloc[0]
    assert str(dominant["track_id"]) == "good-2"


def test_candidate_mixture_anchor_pulls_toward_initial_estimates() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "offset",
                "x_m": 10.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            }
        ]
    )
    initial = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "state_x_m": 0.0,
                "state_y_m": 0.0,
                "state_z_m": 0.0,
            }
        ]
    )

    unanchored = run_candidate_mixture_map(
        candidates,
        initial_estimates=initial,
        config=CandidateMixtureMapConfig(
            top_k=1,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=0.0,
            anchor_weight=0.0,
            iterations=1,
        ),
    )
    anchored = run_candidate_mixture_map(
        candidates,
        initial_estimates=initial,
        config=CandidateMixtureMapConfig(
            top_k=1,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=0.0,
            anchor_weight=9.0,
            iterations=1,
        ),
    )

    assert unanchored.estimates["state_x_m"].iloc[0] == pytest.approx(10.0)
    assert anchored.estimates["state_x_m"].iloc[0] == pytest.approx(1.0)
    assert anchored.summary["config"]["anchor_weight"] == pytest.approx(9.0)


def test_candidate_mixture_can_estimate_on_target_template_times() -> None:
    candidates = pd.DataFrame(
        [
            {
                "sequence_id": "seqA",
                "time_s": 0.0,
                "source": "lidar_360",
                "track_id": "left",
                "x_m": 0.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
            {
                "sequence_id": "seqA",
                "time_s": 2.0,
                "source": "lidar_360",
                "track_id": "right",
                "x_m": 2.0,
                "y_m": 0.0,
                "z_m": 0.0,
                "ranker_score": 1.0,
                "predicted_sigma_m": 1.0,
            },
        ]
    )
    template = pd.DataFrame(
        {
            "Sequence": ["seqA", "seqA", "seqA"],
            "Timestamp": [0.0, 1.0, 2.0],
        }
    )

    result = run_candidate_mixture_map(
        candidates,
        target_template=template,
        config=CandidateMixtureMapConfig(
            top_k=1,
            score_column="ranker_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=0.0,
            target_time_tolerance_s=0.1,
            iterations=1,
        ),
    )

    assert result.estimates["time_s"].tolist() == [0.0, 1.0, 2.0]
    assert len(result.assignments) == 3
    middle = result.assignments.loc[result.assignments["time_s"] == 1.0].iloc[0]
    assert str(middle["track_id"]) == "left"


def test_candidate_mixture_cli_writes_diagnostics(tmp_path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    template_csv = tmp_path / "template.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    official_results_csv = tmp_path / "mmaud_results.csv"
    official_zip = tmp_path / "ug2_submission.zip"
    _uncertainty_candidates().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)
    pd.DataFrame(
        {
            "Sequence": ["seqA"] * 5,
            "Timestamp": np.arange(5, dtype=float),
            "Position": [""] * 5,
            "Classification": [""] * 5,
        }
    ).to_csv(template_csv, index=False)
    pd.DataFrame([{"sequence_id": "seqA", "uav_type": 3}]).to_csv(
        class_map_csv,
        index=False,
    )

    status = mixture_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--target-template-csv",
            str(template_csv),
            "--truth-csv",
            str(truth_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "2",
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--smoothness-weight",
            "100",
            "--anchor-weight",
            "0.01",
            "--iterations",
            "5",
            "--class-map",
            str(class_map_csv),
            "--official-results-csv",
            str(official_results_csv),
            "--official-zip",
            str(official_zip),
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_assignments.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_iterations.csv").exists()
    summary = json.loads(
        (output_dir / "mmuad_candidate_mixture_summary.json").read_text(encoding="utf-8")
    )
    assert summary["metrics"]["pooled"]["rmse_3d_m"] < 0.05
    assert summary["config"]["anchor_weight"] == pytest.approx(0.01)
    official = pd.read_csv(official_results_csv)
    assert official["Classification"].tolist() == [3, 3, 3, 3, 3]
    with ZipFile(official_zip) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_candidate_mixture_validates_uniform_weight_floor() -> None:
    with pytest.raises(ValueError, match="uniform_weight_floor"):
        run_candidate_mixture_map(
            _uncertainty_candidates(),
            config=CandidateMixtureMapConfig(uniform_weight_floor=1.0),
        )


def test_candidate_mixture_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-candidate-mixture-map"]
        == "raft_uav.mmuad.candidate_mixture_map:main"
    )
