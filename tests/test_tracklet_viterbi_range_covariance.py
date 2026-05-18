from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.baselines.tracklet_viterbi import TrackletViterbiAssociationConfig
from raft_uav.baselines.tracklet_viterbi_range_covariance import (
    _radar_row_covariance,
    _write_radar_covariance_diagnostics,
)


def test_range_adaptive_radar_covariance_inflates_long_range_rows() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    row = pd.Series({"range_m": 1000.0})

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.isclose(np.sqrt(covariance[0, 0]), 35.0)
    assert np.isclose(np.sqrt(covariance[1, 1]), 35.0)
    assert np.isclose(np.sqrt(covariance[2, 2]), 50.0)


def test_range_adaptive_radar_covariance_keeps_default_as_lower_bound() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    config = TrackletViterbiAssociationConfig(range_gate_m=None)
    row = pd.Series({"range_m": 100.0})

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.allclose(covariance, default_covariance)


def test_range_adaptive_radar_covariance_can_be_disabled() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    config = _Config(use_range_adaptive_radar_covariance=False)
    row = pd.Series({"range_m": 1000.0})

    covariance = _radar_row_covariance(row, default_covariance, config)

    assert np.allclose(covariance, default_covariance)


def test_radar_covariance_diagnostics_mark_adaptive_rows() -> None:
    default_covariance = np.diag([25.0**2, 25.0**2, 35.0**2])
    row_covariance = np.diag([40.0**2, 40.0**2, 55.0**2])
    row = pd.Series({"range_m": 1200.0})

    _write_radar_covariance_diagnostics(row, row_covariance, default_covariance)

    assert float(row["association_radar_xy_std_m"]) == 40.0
    assert float(row["association_radar_z_std_m"]) == 55.0
    assert bool(row["association_radar_covariance_adaptive"])


class _Config:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)
