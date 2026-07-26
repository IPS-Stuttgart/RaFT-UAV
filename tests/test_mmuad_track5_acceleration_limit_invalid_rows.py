from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.track5_acceleration_limit import repair_track5_acceleration_kinks


def _official_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq0001", "seq0001", "seq0001"],
            "Timestamp": [0.0, "invalid", 2.0],
            "Position": ["(0, 0, 0)", "(10, 0, 0)", "(2, 0, 0)"],
            "Classification": [2, 2, 2],
        }
    )


def test_acceleration_limit_rejects_invalid_official_rows_instead_of_dropping_them() -> None:
    with pytest.raises(ValueError, match=r"row indices: 1$"):
        repair_track5_acceleration_kinks(_official_submission())
