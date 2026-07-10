from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from raft_uav.mmuad.candidate_mixture_map import main as mixture_main


def _write_candidates(path: Path, *, rows: list[tuple[float, float]]) -> None:
    records = [
        {
            "sequence_id": "001",
            "time_s": time_s,
            "source": "lidar_360",
            "track_id": f"candidate-{index}",
            "candidate_branch": "raw",
            "x_m": x_m,
            "y_m": 0.0,
            "z_m": 0.0,
            "ranker_score": 1.0,
            "predicted_sigma_m": 1.0,
        }
        for index, (time_s, x_m) in enumerate(rows)
    ]
    pd.DataFrame.from_records(records).to_csv(path, index=False)


def test_candidate_mixture_module_cli_preserves_target_template_sequence_id(
    tmp_path: Path,
) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    template_csv = tmp_path / "template.csv"
    output_dir = tmp_path / "out"
    _write_candidates(candidates_csv, rows=[(0.0, 0.0), (1.0, 1.0)])
    template_csv.write_text(
        "Sequence,Timestamp\n"
        "001,0.0\n"
        "001,1.0\n",
        encoding="utf-8",
    )

    status = mixture_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--target-template-csv",
            str(template_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--smoothness-weight",
            "0",
            "--iterations",
            "1",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(
        output_dir / "mmuad_candidate_mixture_estimates.csv",
        dtype={"sequence_id": "string"},
    )
    assert estimates["sequence_id"].tolist() == ["001", "001"]
    assert estimates["time_s"].tolist() == pytest.approx([0.0, 1.0])


def test_candidate_mixture_module_cli_preserves_initial_estimate_sequence_id(
    tmp_path: Path,
) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    initial_csv = tmp_path / "initial.csv"
    output_dir = tmp_path / "out"
    _write_candidates(candidates_csv, rows=[(0.0, 10.0)])
    initial_csv.write_text(
        "sequence_id,time_s,state_x_m,state_y_m,state_z_m\n"
        "001,0.0,0.0,0.0,0.0\n",
        encoding="utf-8",
    )

    status = mixture_main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--initial-estimates-csv",
            str(initial_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--smoothness-weight",
            "0",
            "--anchor-weight",
            "9",
            "--iterations",
            "1",
        ]
    )

    assert status == 0
    estimates = pd.read_csv(
        output_dir / "mmuad_candidate_mixture_estimates.csv",
        dtype={"sequence_id": "string"},
    )
    assert estimates.loc[0, "sequence_id"] == "001"
    assert estimates.loc[0, "state_x_m"] == pytest.approx(1.0, abs=1.0e-6)
