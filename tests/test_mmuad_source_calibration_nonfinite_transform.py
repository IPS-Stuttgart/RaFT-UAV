from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import (
    SOURCE_CALIBRATION_SCHEMA,
    apply_source_calibration_json,
)


def _candidate_frame() -> CandidateFrame:
    return CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq001"],
                "time_s": [0.0],
                "source": ["lidar_360"],
                "track_id": ["candidate"],
                "x_m": [10.0],
                "y_m": [20.0],
                "z_m": [30.0],
                "confidence": [1.0],
            }
        )
    )


@pytest.mark.parametrize(
    ("transform", "error_message"),
    [
        (
            {
                "linear": [
                    [1.0, 0.0, 0.0],
                    [0.0, float("nan"), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "translation_m": [0.0, 0.0, 0.0],
            },
            "linear transform must contain only finite values",
        ),
        (
            {
                "linear": np.eye(3).tolist(),
                "translation_m": [0.0, float("inf"), 0.0],
            },
            "translation_m must contain only finite values",
        ),
    ],
)
def test_apply_source_calibration_rejects_nonfinite_transform_json(
    tmp_path: Path,
    transform: dict[str, object],
    error_message: str,
) -> None:
    payload = {
        "schema": SOURCE_CALIBRATION_SCHEMA,
        "mode": "source-translation",
        "transforms": {"lidar_360": transform},
    }
    calibration_json = tmp_path / "invalid_source_calibration.json"
    calibration_json.write_text(
        json.dumps(payload, allow_nan=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=error_message):
        apply_source_calibration_json(_candidate_frame(), calibration_json)
