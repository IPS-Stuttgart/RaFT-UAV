from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from raft_uav.mmuad.track5_submission_ensemble import write_track5_submission_ensemble_outputs


def test_track5_submission_ensemble_outputs_preserve_row_level_classifications(
    tmp_path: Path,
) -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "source": ["track5-submission-ensemble", "track5-submission-ensemble"],
            "track_id": ["track5-submission-ensemble", "track5-submission-ensemble"],
            "state_x_m": [0.0, 1.0],
            "state_y_m": [0.0, 1.0],
            "state_z_m": [0.0, 1.0],
            "Classification": [1, 3],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001"],
            "time_s": [0.0, 1.0],
            "position_spread_m": [0.0, 0.0],
        }
    )

    paths = write_track5_submission_ensemble_outputs(
        estimates=estimates,
        diagnostics=diagnostics,
        output_dir=tmp_path / "out",
    )

    csv_frame = pd.read_csv(paths["results_csv"])
    assert csv_frame["Classification"].tolist() == [1, 3]

    with ZipFile(paths["zip"]) as archive:
        zipped_frame = pd.read_csv(archive.open("mmaud_results.csv"))
    assert zipped_frame["Classification"].tolist() == [1, 3]
