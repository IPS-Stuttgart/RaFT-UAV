from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def _submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sequence_id": ["seq0001", "seq0001", "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 10.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )


@pytest.mark.parametrize("value", [None, pd.NA, np.nan, "", "   "])
def test_acceleration_limit_rejects_missing_or_blank_sequence_ids(
    value: object,
) -> None:
    rows = _submission()
    rows.loc[1, "sequence_id"] = value

    with pytest.raises(ValueError) as error:
        repair_track5_acceleration_kinks(rows)

    message = str(error.value)
    assert "submission contains missing or blank sequence_id values" in message
    assert "sequence_id rows [1]" in message


def test_acceleration_limit_rejects_missing_categorical_sequence_ids() -> None:
    rows = _submission()
    rows["sequence_id"] = pd.Categorical(
        ["seq0001", None, "seq0001"],
        categories=["seq0001"],
    )

    with pytest.raises(ValueError, match=r"sequence_id rows \[1\]"):
        repair_track5_acceleration_kinks(rows)


def test_acceleration_limit_keeps_nonblank_numeric_sequence_ids() -> None:
    rows = _submission()
    rows["sequence_id"] = [0, 0, 0]

    repaired, diagnostics = repair_track5_acceleration_kinks(rows, iterations=1)

    assert repaired["sequence_id"].tolist() == ["0", "0", "0"]
    assert diagnostics["sequence_id"].tolist() == ["0", "0", "0"]


def test_acceleration_limit_rejects_duplicate_keys_after_sequence_id_canonicalization() -> None:
    rows = _submission().iloc[:2].copy()
    rows["sequence_id"] = ["seq0001", " seq0001 "]
    rows["time_s"] = [0.0, 0.0]

    with pytest.raises(ValueError, match=r"duplicate .*seq0001@0"):
        repair_track5_acceleration_kinks(rows, iterations=1)


def test_acceleration_limit_returns_canonical_sequence_ids() -> None:
    rows = _submission().iloc[:2].copy()
    rows["sequence_id"] = [" seq0001 ", "seq0001"]

    repaired, diagnostics = repair_track5_acceleration_kinks(rows, iterations=1)

    assert repaired["sequence_id"].tolist() == ["seq0001", "seq0001"]
    assert diagnostics["sequence_id"].tolist() == ["seq0001", "seq0001"]
