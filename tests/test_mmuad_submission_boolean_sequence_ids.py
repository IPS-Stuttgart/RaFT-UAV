from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.submission import parse_official_sequence_cell
from raft_uav.mmuad.track5_speed_limit import project_track5_speed_limit


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_official_sequence_parser_rejects_boolean_values(value: object) -> None:
    with pytest.raises(ValueError, match="identifiers, not booleans"):
        parse_official_sequence_cell(value)


def test_official_sequence_parser_preserves_supported_identifiers() -> None:
    assert parse_official_sequence_cell(" 001 ") == "001"
    assert parse_official_sequence_cell(7) == "7"


@pytest.mark.parametrize("value", [True, False])
def test_speed_limit_rejects_boolean_sequence_identifiers(value: bool) -> None:
    rows = pd.DataFrame(
        {
            "sequence_id": ["seq0001", value, "seq0001"],
            "time_s": [0.0, 1.0, 2.0],
            "state_x_m": [0.0, 1.0, 2.0],
            "state_y_m": [0.0, 0.0, 0.0],
            "state_z_m": [0.0, 0.0, 0.0],
            "Classification": [2, 2, 2],
        }
    )

    with pytest.raises(ValueError, match="sequence_id"):
        project_track5_speed_limit(rows)
