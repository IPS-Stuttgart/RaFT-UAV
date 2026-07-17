from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.submission import validate_official_track5_submission


def _submission_and_template(tmp_path):
    path = tmp_path / "mmaud_results.csv"
    frame = pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(1,2,3)"],
            "Classification": [1],
        }
    )
    frame.to_csv(path, index=False)
    return path, frame


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        np.array([0.25]),
        0.25 + 0.0j,
    ],
)
def test_submission_validator_rejects_invalid_timestamp_tolerances(
    tmp_path,
    value: object,
) -> None:
    path, template = _submission_and_template(tmp_path)

    with pytest.raises(
        ValueError,
        match="timestamp_tolerance_s must be non-negative and finite",
    ):
        validate_official_track5_submission(
            path,
            template=template,
            timestamp_tolerance_s=value,
            require_zip=False,
        )


@pytest.mark.parametrize("value", [0.0, "0.25", np.float64(0.25), np.array(0.25)])
def test_submission_validator_accepts_real_scalar_timestamp_tolerances(
    tmp_path,
    value: object,
) -> None:
    path, template = _submission_and_template(tmp_path)

    validation = validate_official_track5_submission(
        path,
        template=template,
        timestamp_tolerance_s=value,
        require_zip=False,
    )

    assert validation.summary["timestamp_tolerance_s"] == pytest.approx(float(value))
