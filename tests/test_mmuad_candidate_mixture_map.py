from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureConfig,
    compute_candidate_responsibilities,
    main,
    run_candidate_mixture_map,
)


def _candidate(
    *,
    time_s: float,
    x_m: float,
    score: float = 1.0,
    branch: str = "raw",
    source: str = "lidar_360",
    sigma_m: float = 1.0,
) -> dict[str, object]:
    return {
        "sequence_id": "seq0001",
        "time_s": time_s,
        "source": source,
        "track_id": f"{branch}-{time_s}-{x_m}",
        "candidate_branch": branch,
        "x_m": x_m,
        "y_m": 0.0,
        "z_m": 0.0,
        "candidate_reservoir_score": score,
        "predicted_sigma_m": sigma_m,
        "std_xy_m": sigma_m,
        "std_z_m": sigma_m,
        "confidence": score,
    }


def test_branch_balance_preserves_mass_for_small_candidate_branch() -> None:
    rows = pd.DataFrame(
        [
            _candidate(time_s=0.0, x_m=0.0, branch="raw"),
            _candidate(time_s=0.0, x_m=0.5, branch="raw"),
            _candidate(time_s=0.0, x_m=10.0, branch="translated"),
        ]
    )
    responsibilities = compute_candidate_responsibilities(
        rows,
        np.asarray([0.0, 0.0, 0.0]),
        config=CandidateMixtureConfig(
            top_k=0,
            huber_scale=0.0,
            branch_balance=1.0,
            source_balance=0.0,
            responsibility_floor=0.0,
        ),
    )

    mass = responsibilities.groupby("candidate_branch")["mixture_responsibility"].sum()
    assert mass["raw"] == pytest.approx(0.5)
    assert mass["translated"] == pytest.approx(0.5)


def test_huber_mixture_rejects_far_low_score_candidates() -> None:
    rows: list[dict[str, object]] = []
    for time_s, correct_x in enumerate((0.0, 1.0, 2.0)):
        rows.append(_candidate(time_s=float(time_s), x_m=correct_x, score=2.0))
        rows.append(
            _candidate(
                time_s=float(time_s),
                x_m=100.0,
                score=-2.0,
                branch="clutter",
                source="dynamic",
            )
        )
    result = run_candidate_mixture_map(
        pd.DataFrame(rows),
        config=CandidateMixtureConfig(
            top_k=0,
            temperature=1.0,
            smoothness_weight=10.0,
            huber_scale=1.0,
            iterations=4,
            branch_balance=0.0,
            responsibility_floor=0.0,
            initialization="best-score",
        ),
    )

    expected = np.asarray([0.0, 1.0, 2.0])
    actual = result.estimates["state_x_m"].to_numpy(float)
    assert np.sqrt(np.mean((actual - expected) ** 2)) < 1.0
    assert result.frame_diagnostics["dominant_candidate_branch"].eq("raw").all()


def test_acceleration_regularization_reduces_isolated_impulse() -> None:
    rows = pd.DataFrame(
        [
            _candidate(time_s=0.0, x_m=0.0),
            _candidate(time_s=1.0, x_m=20.0),
            _candidate(time_s=2.0, x_m=0.0),
        ]
    )
    result = run_candidate_mixture_map(
        rows,
        config=CandidateMixtureConfig(
            top_k=0,
            smoothness_weight=100.0,
            huber_scale=0.0,
            iterations=1,
            branch_balance=0.0,
            responsibility_floor=0.0,
            measurement_weight_mode="uniform",
        ),
    )

    middle = float(result.estimates.loc[1, "state_x_m"])
    assert 0.0 < middle < 20.0
    assert result.iteration_summary.loc[0, "mean_state_change_m"] > 0.0


def test_cli_writes_estimates_diagnostics_and_official_zip(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        [
            _candidate(time_s=0.0, x_m=0.0),
            _candidate(time_s=1.0, x_m=1.0),
        ]
    )
    candidates_csv = tmp_path / "candidates.csv"
    candidates.to_csv(candidates_csv, index=False)
    class_map = tmp_path / "class_map.csv"
    pd.DataFrame([{"sequence_id": "seq0001", "uav_type": 3}]).to_csv(
        class_map,
        index=False,
    )
    estimates_csv = tmp_path / "estimates.csv"
    diagnostics_csv = tmp_path / "diagnostics.csv"
    assignments_csv = tmp_path / "assignments.csv"
    summary_json = tmp_path / "summary.json"
    results_csv = tmp_path / "mmaud_results.csv"
    official_zip = tmp_path / "ug2_submission.zip"

    rc = main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--output-estimates-csv",
            str(estimates_csv),
            "--frame-diagnostics-csv",
            str(diagnostics_csv),
            "--candidate-assignments-csv",
            str(assignments_csv),
            "--summary-json",
            str(summary_json),
            "--top-k",
            "0",
            "--iterations",
            "1",
            "--smoothness-weight",
            "0",
            "--branch-balance",
            "0",
            "--responsibility-floor",
            "0",
            "--class-map",
            str(class_map),
            "--official-results-csv",
            str(results_csv),
            "--official-zip",
            str(official_zip),
        ]
    )

    assert rc == 0
    assert estimates_csv.exists()
    assert diagnostics_csv.exists()
    assert assignments_csv.exists()
    assert summary_json.exists()
    official = pd.read_csv(results_csv)
    assert official["Classification"].tolist() == [3, 3]
    with ZipFile(official_zip) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
