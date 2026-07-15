from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_template_resample import (
    resample_estimates_to_track5_template,
)


def test_nearest_classification_uses_same_duplicate_as_position() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 0.0, 10.0],
            "state_x_m": [1.0, 2.0, 10.0],
            "state_y_m": [11.0, 12.0, 20.0],
            "state_z_m": [21.0, 22.0, 30.0],
            "classification": [1, 2, 3],
        }
    )
    template = pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [0.0]})

    resampled, _ = resample_estimates_to_track5_template(
        estimates,
        template,
        resample_method="nearest",
        classification_policy="nearest",
    )

    row = resampled.iloc[0]
    assert row["state_x_m"] == pytest.approx(2.0)
    assert row["state_y_m"] == pytest.approx(12.0)
    assert row["state_z_m"] == pytest.approx(22.0)
    assert row["classification"] == 2


def test_duplicate_timestamps_do_not_overweight_sequence_mode() -> None:
    estimates = pd.DataFrame(
        {
            "sequence_id": ["seq0001"] * 4,
            "time_s": [0.0, 0.0, 0.0, 10.0],
            "state_x_m": [0.0, 1.0, 2.0, 10.0],
            "state_y_m": [0.0, 1.0, 2.0, 10.0],
            "state_z_m": [0.0, 1.0, 2.0, 10.0],
            "classification": [1, 1, 2, 2],
        }
    )
    template = pd.DataFrame({"Sequence": ["seq0001"], "Timestamp": [5.0]})

    resampled, _ = resample_estimates_to_track5_template(
        estimates,
        template,
        classification_policy="sequence-mode",
    )

    row = resampled.iloc[0]
    assert row["state_x_m"] == pytest.approx(6.0)
    assert row["state_y_m"] == pytest.approx(6.0)
    assert row["state_z_m"] == pytest.approx(6.0)
    assert row["classification"] == 2
