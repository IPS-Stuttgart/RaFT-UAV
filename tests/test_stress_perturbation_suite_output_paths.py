from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from raft_uav.experiments.stress_perturbation_suite import main


def test_cli_rejects_perturbation_names_with_colliding_output_paths(
    tmp_path: Path,
) -> None:
    input_csv = tmp_path / "measurements.csv"
    output_dir = tmp_path / "outputs"
    pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [1.0],
            "north_m": [2.0],
            "up_m": [3.0],
        }
    ).to_csv(input_csv, index=False)

    with pytest.raises(ValueError, match="map to unique output filenames") as error:
        main(
            [
                str(input_csv),
                "--output-dir",
                str(output_dir),
                "--spec",
                json.dumps({"name": "camera/drop"}),
                "--spec",
                json.dumps({"name": "camera_drop"}),
            ]
        )

    message = str(error.value)
    assert "'camera/drop'" in message
    assert "'camera_drop'" in message
    assert "'camera_drop.csv'" in message
    assert not output_dir.exists()
