import json
from pathlib import Path

import numpy as np
import pandas as pd

from raft_uav.calibration.lofo_io import jsonable


def test_jsonable_normalizes_numpy_arrays_and_pandas_missing(tmp_path: Path) -> None:
    payload = {
        "coefficients": np.array([1.5, np.nan, np.inf]),
        "object_values": np.array([pd.NA, np.int64(4)], dtype=object),
        "missing": pd.NA,
        "not_a_time": pd.NaT,
        "model_path": tmp_path / "bias_model.json",
    }

    normalized = jsonable(payload)

    assert normalized == {
        "coefficients": [1.5, None, None],
        "object_values": [None, 4],
        "missing": None,
        "not_a_time": None,
        "model_path": str(tmp_path / "bias_model.json"),
    }
    assert json.loads(json.dumps(normalized, allow_nan=False)) == normalized
