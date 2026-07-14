from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import CandidateFrame
from raft_uav.mmuad.source_calibration import fit_source_calibration
from raft_uav.mmuad.source_calibration import parse_source_translation_alpha_grid


@pytest.mark.parametrize(
    "value",
    [
        "-0.1",
        "1.1",
        "nan,0.5",
        "inf",
        True,
    ],
)
def test_source_calibration_rejects_lossy_alpha_grid_values(value: object) -> None:
    with pytest.raises(ValueError, match="source_translation_alpha_grid.*\[0, 1\]"):
        parse_source_translation_alpha_grid(value)


def test_source_calibration_rejects_explicit_empty_alpha_grid() -> None:
    candidates = CandidateFrame(
        pd.DataFrame(
            {
                "sequence_id": ["seq001"],
                "time_s": [0.0],
                "source": ["lidar_360"],
                "track_id": ["candidate"],
                "x_m": [1.0],
                "y_m": [0.0],
                "z_m": [0.0],
                "confidence": [1.0],
            }
        )
    )
    truth = pd.DataFrame(
        {
            "sequence_id": ["seq001"],
            "time_s": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
            "z_m": [0.0],
        }
    )

    with pytest.raises(ValueError, match="must contain at least one value"):
        fit_source_calibration(
            candidates,
            truth,
            mode="source-translation",
            min_pairs_per_source=1,
            source_translation_alpha_grid=[],
        )


def test_source_calibration_keeps_valid_alpha_grid_exact() -> None:
    parsed = parse_source_translation_alpha_grid("1,0.5,0,0.5")

    assert parsed == (0.0, 0.5, 1.0)
    assert all(np.isfinite(parsed))
