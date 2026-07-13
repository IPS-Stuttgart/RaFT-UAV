from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.mmuad.template_snap_write import _jsonable


def test_template_snap_jsonable_normalizes_nonfinite_and_missing_scalars() -> None:
    payload = {
        "finite": 1.25,
        "nan": float("nan"),
        "positive_infinity": np.float64(np.inf),
        "negative_infinity": np.float32(-np.inf),
        "missing": pd.NA,
        "path": Path("outputs/result.json"),
        "nested": [np.int64(3), np.float64(np.nan)],
    }

    text = json.dumps(_jsonable(payload), allow_nan=False)
    decoded = json.loads(text)

    assert decoded == {
        "finite": 1.25,
        "nan": None,
        "positive_infinity": None,
        "negative_infinity": None,
        "missing": None,
        "path": "outputs/result.json",
        "nested": [3, None],
    }
