from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.track5_estimate_sequence_gate import _sequence_weight_map


@pytest.mark.parametrize("value", [None, np.nan, "", "   "])
def test_sequence_weight_map_rejects_invalid_sequence_identifiers(
    value: object,
) -> None:
    weights = pd.DataFrame(
        {
            "sequence_id": ["seq0001", value],
            "weight": [0.25, 0.75],
        },
        index=[10, 42],
    )

    with pytest.raises(
        ValueError,
        match=r"invalid sequence identifier at row 42",
    ):
        _sequence_weight_map(weights)


def test_sequence_weight_map_preserves_valid_opaque_identifiers() -> None:
    weights = pd.DataFrame(
        {
            "Sequence": ["001", "seq0002"],
            "blend_weight": ["0.25", "0.75"],
        }
    )

    assert _sequence_weight_map(weights) == {
        "001": 0.25,
        "seq0002": 0.75,
    }
