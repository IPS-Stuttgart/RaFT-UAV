from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from raft_uav.mmuad import stratified_mixture_submission


def _candidates() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for time_s in range(3):
        records.append(
            {
                "sequence_id": "seqA",
                "time_s": float(time_s),
                "source": "lidar_360",
                "track_id": f"raw-good-{time_s}",
                "candidate_branch": "raw",
                "x_m": float(time_s),
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.10,
                "predicted_sigma_m": 1.0,
            }
        )
        records.append(
            {
                "sequence_id": "seqA",
                "time_s": float(time_s),
                "source": "lidar_360",
                "track_id": f"translated-bad-{time_s}",
                "candidate_branch": "translated",
                "x_m": float(time_s + 20),
                "y_m": 0.0,
                "z_m": 1.0,
                "ranker_score": 0.99,
                "predicted_sigma_m": 20.0,
            }
        )
    return pd.DataFrame.from_records(records)


def _truth() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seqA"] * 3,
            "time_s": np.arange(3, dtype=float),
            "x_m": np.arange(3, dtype=float),
            "y_m": np.zeros(3),
            "z_m": np.ones(3),
        }
    )


def test_installed_stratified_submission_module_writes_official_zip(tmp_path: Path) -> None:
    candidates_csv = tmp_path / "candidates.csv"
    truth_csv = tmp_path / "truth.csv"
    class_map_csv = tmp_path / "class_map.csv"
    output_dir = tmp_path / "out"
    _candidates().to_csv(candidates_csv, index=False)
    _truth().to_csv(truth_csv, index=False)
    pd.DataFrame([{"sequence_id": "seqA", "uav_type": 3}]).to_csv(
        class_map_csv,
        index=False,
    )

    status = stratified_mixture_submission.main(
        [
            "--candidates-csv",
            str(candidates_csv),
            "--truth-csv",
            str(truth_csv),
            "--class-map",
            str(class_map_csv),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "2",
            "--min-per-branch",
            "1",
            "--min-per-source",
            "0",
            "--score-column",
            "ranker_score",
            "--sigma-column",
            "predicted_sigma_m",
            "--smoothness-weight",
            "100",
            "--iterations",
            "5",
        ]
    )

    assert status == 0
    official_results = output_dir / "mmaud_results.csv"
    official_zip = output_dir / "ug2_submission.zip"
    assert official_results.exists()
    assert official_zip.exists()
    official = pd.read_csv(official_results)
    assert official["Classification"].tolist() == [3, 3, 3]
    with ZipFile(official_zip) as archive:
        assert archive.namelist() == ["mmaud_results.csv"]
