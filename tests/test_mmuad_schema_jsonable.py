import json

import numpy as np
import pandas as pd

from raft_uav.mmuad.schema import load_jsonable


def test_load_jsonable_converts_pandas_missing_scalars_to_json_null():
    payload = load_jsonable(
        {
            "pd_na": pd.NA,
            "pd_nat": pd.NaT,
            "float_nan": float("nan"),
            "numpy_nan": np.float64(np.nan),
            "nested": [1, pd.NA, {"nat": pd.NaT}],
        }
    )

    expected = {
        "pd_na": None,
        "pd_nat": None,
        "float_nan": None,
        "numpy_nan": None,
        "nested": [1, None, {"nat": None}],
    }
    assert payload == expected
    assert json.loads(json.dumps(payload)) == expected
