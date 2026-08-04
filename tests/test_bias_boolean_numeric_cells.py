from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.calibration.bias import (
    _finite_real_numeric_series,
    make_bias_training_examples,
)


def _nested_boolean(value: bool = True) -> np.ndarray:
    outer = np.empty((), dtype=object)
    outer[()] = np.asarray(value)
    return outer


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(False), np.asarray(True), _nested_boolean()],
)
def test_bias_numeric_normalization_rejects_boolean_cells(value: object) -> None:
    normalized = _finite_real_numeric_series(pd.Series([value]))

    assert normalized.isna().tolist() == [True]


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("measurements", "time_s", True),
        ("measurements", "east_m", np.bool_(False)),
        ("measurements", "north_m", np.asarray(True)),
        ("truth", "time_s", _nested_boolean(False)),
        ("truth", "east_m", True),
        ("truth", "north_m", np.asarray(False)),
    ],
)
def test_bias_training_drops_rows_with_boolean_numeric_cells(
    table: str,
    column: str,
    value: object,
) -> None:
    measurements = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [10.0, 20.0],
            "north_m": [30.0, 40.0],
        }
    )
    truth = pd.DataFrame(
        {
            "time_s": [0.0, 1.0],
            "east_m": [9.0, 19.0],
            "north_m": [29.0, 39.0],
        }
    )
    frame = measurements if table == "measurements" else truth
    frame[column] = frame[column].astype(object)
    frame.at[1, column] = value

    rows = make_bias_training_examples(
        measurements,
        truth,
        source="rf",
        target_columns=("east_m", "north_m"),
        time_gate_s=0.0,
    )

    assert rows["time_s"].tolist() == [0.0]
    assert rows["bias_east_m"].tolist() == [1.0]
    assert rows["bias_north_m"].tolist() == [1.0]
