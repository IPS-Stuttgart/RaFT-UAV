from __future__ import annotations

import json
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_pull import CandidatePullConfig
from raft_uav.mmuad.candidate_pull import candidate_pull_input_provenance
from raft_uav.mmuad.candidate_pull import assign_candidate_pull_alphas
from raft_uav.mmuad.candidate_pull import main as candidate_pull_main
from raft_uav.mmuad.candidate_pull import parse_position
from raft_uav.mmuad.candidate_pull import refine_official_results_with_candidate_pull


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001"],
            "Timestamp": [0.0, 1.0],
            "Position": ["(0,0,0)", "(2,0,0)"],
            "Classification": [2, 2],
        }
    )


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, 0.0, 1.0, 1.0],
            "x_m": [10.0, 99.0, 4.0, 99.0],
            "y_m": [2.0, 99.0, 4.0, 99.0],
            "z_m": [6.0, 99.0, 8.0, 99.0],
            "ranker_score": [0.9, 0.1, 0.8, 0.1],
            "confidence": [0.8, 0.1, 0.8, 0.1],
            "cluster_point_count": [20, 1, 20, 1],
            "nearest_cross_sensor_distance_m": [1.0, 10.0, 2.0, 10.0],
            "cross_sensor_neighbor_count": [2, 0, 1, 0],
            "frame_source_count": [2, 1, 2, 1],
        }
    )


def test_candidate_pull_constant_policy_preserves_classes_and_pulls_position() -> None:
    result = refine_official_results_with_candidate_pull(
        _results(),
        _candidates(),
        config=CandidatePullConfig(
            policy="constant",
            smoother="none",
            constant_alpha_xy=0.5,
            constant_alpha_z=0.25,
            top_k=1,
        ),
    )

    first_xyz = parse_position(result.rows.iloc[0]["Position"])
    second_xyz = parse_position(result.rows.iloc[1]["Position"])

    assert first_xyz.tolist() == pytest.approx([5.0, 1.0, 1.5])
    assert second_xyz.tolist() == pytest.approx([3.0, 2.0, 2.0])
    assert result.rows["Classification"].tolist() == [2, 2]
    assert result.provenance["matched_candidate_center_count"] == 2


def test_candidate_pull_feature_rule_v2_uses_sequence_features() -> None:
    sequence_features = pd.DataFrame(
        {
            "Sequence": ["compact", "gap"],
            "top_score_mean": [0.64, 0.9],
            "dispersion5_mean": [0.05, 2.0],
            "current_distance_mean": [1.0, 25.0],
        }
    )

    alphas = assign_candidate_pull_alphas(sequence_features, policy="feature-rule-v2")

    compact = alphas.set_index("Sequence").loc["compact"]
    gap = alphas.set_index("Sequence").loc["gap"]
    assert compact["candidate_pull_alpha_xy"] == pytest.approx(-0.5)
    assert compact["candidate_pull_reason"] == "ultra_compact_low_score"
    assert gap["candidate_pull_alpha_xy"] == pytest.approx(1.2)
    assert gap["candidate_pull_reason"] == "large_current_to_candidate_gap"


def test_candidate_pull_cli_writes_zip_and_provenance_without_truth(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    candidates_csv = tmp_path / "candidates.csv"
    out_csv = tmp_path / "out" / "mmaud_results.csv"
    zip_path = tmp_path / "out" / "ug2_submission.zip"
    provenance_path = tmp_path / "out" / "candidate_pull_provenance.json"
    _results().to_csv(results_csv, index=False)
    _candidates().to_csv(candidates_csv, index=False)

    status = candidate_pull_main(
        [
            "--results-in",
            str(results_csv),
            "--candidates",
            str(candidates_csv),
            "--results-out",
            str(out_csv),
            "--submission-zip",
            str(zip_path),
            "--provenance-json",
            str(provenance_path),
            "--candidate-pull-policy",
            "constant",
            "--candidate-pull-alpha-xy",
            "1",
            "--candidate-pull-alpha-z",
            "0",
            "--candidate-pull-top-k",
            "1",
            "--candidate-pull-smoother",
            "none",
        ]
    )

    assert status == 0
    assert out_csv.exists()
    assert zip_path.exists()
    with ZipFile(zip_path) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["schema"] == "raft-uav-mmuad-candidate-pull-provenance-v1"
    assert provenance["policy"] == "constant"
    assert provenance["results_in"] == str(results_csv)
    assert provenance["candidates"] == str(candidates_csv)
    assert provenance["input_result_row_count"] == 2
    assert provenance["input_candidate_row_count"] == 4
    assert provenance["candidate_score_column"] == "ranker_score"
    assert provenance["candidate_score_summary"]["finite_count"] == 4
    assert "truth" not in json.dumps(provenance).lower()


def test_candidate_pull_input_provenance_reports_score_and_sources(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    candidates_csv = tmp_path / "candidates.csv"
    results = _results()
    candidates = _candidates().assign(source=["lidar", "lidar", "radar", "radar"])

    provenance = candidate_pull_input_provenance(
        results_path=results_csv,
        candidates_path=candidates_csv,
        results=results,
        candidates=candidates,
    )

    assert provenance["results_in"] == str(results_csv)
    assert provenance["candidates"] == str(candidates_csv)
    assert provenance["input_candidate_sequence_count"] == 1
    assert provenance["input_candidate_timestamp_count"] == 2
    assert provenance["candidate_score_column"] == "ranker_score"
    assert provenance["candidate_score_summary"]["positive_fraction"] == pytest.approx(1.0)
    assert provenance["candidate_source_counts"] == {"lidar": 2, "radar": 2}


def test_candidate_pull_cli_rejects_truth_argument(tmp_path: Path) -> None:
    results_csv = tmp_path / "results.csv"
    candidates_csv = tmp_path / "candidates.csv"
    _results().to_csv(results_csv, index=False)
    _candidates().to_csv(candidates_csv, index=False)

    with pytest.raises(SystemExit):
        candidate_pull_main(
            [
                "--results-in",
                str(results_csv),
                "--candidates",
                str(candidates_csv),
                "--results-out",
                str(tmp_path / "out.csv"),
                "--truth",
                str(tmp_path / "truth.csv"),
            ]
        )


def test_candidate_pull_entrypoint_is_exposed() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert (
        pyproject["project"]["scripts"]["raft-uav-mmuad-candidate-pull"]
        == "raft_uav.mmuad.candidate_pull:main"
    )
