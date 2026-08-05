from __future__ import annotations

import json

import numpy as np
import pandas as pd

from raft_uav.diagnostics import nis_reliability


def test_nis_reliability_jsonable_normalizes_missing_scalars() -> None:
    payload = {
        "rows": [
            {
                "sequence_id": np.float64(np.nan),
                "optional_label": pd.NA,
                "timestamp": pd.NaT,
                "count": np.int64(2),
            }
        ],
        "quantiles": np.array([0.95, np.nan]),
    }

    normalized = nis_reliability._jsonable(payload)

    assert normalized == {
        "rows": [
            {
                "sequence_id": None,
                "optional_label": None,
                "timestamp": None,
                "count": 2,
            }
        ],
        "quantiles": [0.95, None],
    }
    assert nis_reliability._IMPL._jsonable is nis_reliability._jsonable
    json.dumps(normalized, allow_nan=False)
