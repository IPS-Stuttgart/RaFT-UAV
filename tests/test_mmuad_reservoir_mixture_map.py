from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_reservoir_mixture_map import main as reservoir_mixture_main
from raft_uav.mmuad.candidate_reservoir_mixture_map import run_reservoir_mixture_map


def _branch_candidates() -> pd.DataFrame:
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
                    "x_m": float(time_s + 25.0),
                    "y_m": 0.0,
                    "z_m": 1.0,
                    "ranker_score": 0.99,
                    "predicted_sigma_m": 30.0,
                },
            ]
        )
    return pd.DataFrame.from_records(rows)


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


def test_reservoir_mixture_keeps_low_score_branch_candidate() -> None:
    reservoir, result, summary = run_reservoir_mixture_map(
        _branch_candidates(),
        reservoir_config=ReservoirConfig(
            global_top_n=1,
            per_source_top_n=0,
            per_branch_top_n=1,
            max_candidates_per_frame=2,
            score_column="ranker_score",
            fallback_score_column="confidence",
        ),
        mixture_config=CandidateMixtureMapConfig(
            top_k=0,
            score_column="candidate_reservoir_score",
            sigma_column="predicted_sigma_m",
            smoothness_weight=100.0,
            iterations=5,
        ),
        truth=_truth_rows(),
    )

    assert set(reservoir["candidate_branch"]) == {"raw", "source_translation"}
    assert result.summary["metrics"]["pooled"]["rmse_3d_m"] < 0.1
    dominant = result.assignments.loc[result.assignments["mixture_dominant"]]
    assert dominant["track_id"].astype(str).str.startswith("raw-good-").all()
    assert summary["reservoir"]["reservoir_candidate_rows"] == 10
    assert summary["mixture"]["config"]["top_k"] == 0


def test_reservoir_mixture_cli_writes_upload_artifacts(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    official_csv = tmp_path / "mmaud_results.csv"
    official_zip = tmp_path / "ug2_submission.zip"
    _branch_candidates().to_csv(candidates_csv, index=False)
    _truth_rows().to_csv(truth_csv, index=False)
    pd.DataFrame({"sequence_id": ["seqA"], "uav_type": [2]}).to_csv(
        class_map_csv,
        index=False,
    )

    status = reservoir_mixture_main(
        [
            "--candidate-csv",
            f"union={candidates_csv}",
            "--truth-csv",
            str(truth_csv),
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
            "--class-map",
            str(class_map_csv),
            "--official-results-csv",
            str(official_csv),
            "--official-zip",
            str(official_zip),
        ]
    )

    assert status == 0
    assert (output_dir / "mmuad_reservoir_mixture_candidates.csv").exists()
    assert (output_dir / "mmuad_candidate_mixture_estimates.csv").exists()
    assert (output_dir / "mmuad_reservoir_mixture_summary.json").exists()
    summary = json.loads(
        (output_dir / "mmuad_reservoir_mixture_summary.json").read_text(encoding="utf-8")
    )
    assert summary["mixture"]["metrics"]["pooled"]["rmse_3d_m"] < 0.1
    official = pd.read_csv(official_csv)
    assert official["Classification"].tolist() == [2, 2, 2, 2, 2]
    with ZipFile(official_zip) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]


def test_reservoir_mixture_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-reservoir-mixture-map"]
        == "raft_uav.mmuad.candidate_reservoir_mixture_map:main"
    )
