from __future__ import annotations

import json

import numpy as np
import pandas as pd

from raft_uav.mmuad.track5_submission_ensemble import _jsonable


def test_track5_submission_ensemble_jsonable_emits_strict_missing_values() -> None:
    payload = {
        "python_nan": float("nan"),
        "numpy_nan": np.float64("nan"),
        "numpy_inf": np.float32("inf"),
        "pandas_na": pd.NA,
        "pandas_nat": pd.NaT,
        "masked": np.ma.masked,
        "array": np.array([1.0, np.nan, np.inf]),
        "object_array": np.array(
            [pd.NA, np.float64(-np.inf), np.int64(3)],
            dtype=object,
        ),
    }

    normalized = _jsonable(payload)
    encoded = json.dumps(normalized, allow_nan=False)
    decoded = json.loads(encoded)

    assert decoded == {
        "python_nan": None,
        "numpy_nan": None,
        "numpy_inf": None,
        "pandas_na": None,
        "pandas_nat": None,
        "masked": None,
        "array": [1.0, None, None],
        "object_array": [None, None, 3],
    }
