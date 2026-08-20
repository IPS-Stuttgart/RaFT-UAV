from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.submission import validate_official_track5_submission


def _submission_csv(tmp_path):
    path = tmp_path / "mmaud_results.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(1,2,3)"],
            "Classification": [1],
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    "value",
    ["False", "True", 0, 1, None, [], np.array(True), np.array(False)],
)
def test_submission_validator_rejects_ambiguous_require_zip_values(
    tmp_path,
    value: object,
) -> None:
    path = _submission_csv(tmp_path)

    with pytest.raises(ValueError, match="require_zip must be a Boolean scalar"):
        validate_official_track5_submission(path, require_zip=value)


@pytest.mark.parametrize("value", [False, np.bool_(False)])
def test_submission_validator_accepts_false_boolean_require_zip(
    tmp_path,
    value: object,
) -> None:
    path = _submission_csv(tmp_path)

    validation = validate_official_track5_submission(path, require_zip=value)

    assert validation.summary["require_zip"] is False
    assert validation.summary["is_zip"] is False


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_submission_validator_accepts_true_boolean_require_zip(
    tmp_path,
    value: object,
) -> None:
    path = _submission_csv(tmp_path)

    validation = validate_official_track5_submission(path, require_zip=value)

    assert validation.summary["require_zip"] is True
    assert validation.summary["is_zip"] is False
    assert any("must be a .zip file" in error for error in validation.summary["errors"])
