from __future__ import annotations

import json

import numpy as np

from raft_uav.mmuad.schema import load_jsonable


def test_load_jsonable_converts_longdouble_scalar_to_native_float() -> None:
    result = load_jsonable(np.longdouble("1.25"))

    assert type(result) is float
    assert result == 1.25
    assert json.loads(json.dumps(result, allow_nan=False)) == 1.25


def test_load_jsonable_recursively_normalizes_longdouble_arrays() -> None:
    payload = {
        "values": np.array(
            [np.longdouble("1.25"), np.longdouble("nan"), np.longdouble("2.5")],
            dtype=np.longdouble,
        )
    }

    result = load_jsonable(payload)

    assert result == {"values": [1.25, None, 2.5]}
    assert json.loads(json.dumps(result, allow_nan=False)) == result
