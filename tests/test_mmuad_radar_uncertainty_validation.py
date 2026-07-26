from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.radar import radar_polar_frame_to_candidates


def _radar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "range_m": [10.0],
            "azimuth_deg": [0.0],
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("range_std_m", 0.0),
        ("range_std_m", -1.0),
        ("range_std_m", np.nan),
        ("range_std_m", np.inf),
        ("range_std_m", True),
        ("range_std_m", [2.0]),
        ("angle_std_deg", -1.0),
        ("angle_std_deg", np.nan),
        ("angle_std_deg", False),
        ("angle_std_deg", 1.0 + 0.0j),
        ("z_std_m", 0.0),
        ("z_std_m", -1.0),
        ("z_std_m", np.ma.masked),
    ],
)
def test_radar_adapter_rejects_invalid_uncertainty_parameters(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "range_std_m": 2.0,
        "angle_std_deg": 2.0,
        "z_std_m": 5.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        radar_polar_frame_to_candidates(_radar_frame(), **kwargs)


def test_radar_adapter_accepts_zero_angular_uncertainty() -> None:
    candidates = radar_polar_frame_to_candidates(
        _radar_frame(),
        range_std_m=np.float64(3.0),
        angle_std_deg=np.array(0.0),
        z_std_m=np.int64(4),
    )

    assert candidates.rows["std_xy_m"].tolist() == [3.0]
    assert candidates.rows["std_z_m"].tolist() == [4.0]
