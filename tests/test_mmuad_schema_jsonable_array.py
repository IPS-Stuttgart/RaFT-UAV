from __future__ import annotations

import json

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import load_jsonable


def test_load_jsonable_recurses_into_numpy_array_elements() -> None:
    values = np.array(
        [
            [np.int64(3), np.nan],
            [pd.NA, np.float64(np.inf)],
        ],
        dtype=object,
    )

    converted = load_jsonable(values)

    assert converted == [[3, None], [None, None]]
    assert json.loads(json.dumps(converted, allow_nan=False)) == converted


def test_load_jsonable_normalizes_zero_dimensional_object_array() -> None:
    converted = load_jsonable(np.array(pd.NA, dtype=object))

    assert converted is None
