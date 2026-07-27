from __future__ import annotations

import math

import numpy as np
import pandas as pd

from raft_uav.evaluation.diagnostics import build_diagnostic_summary


def test_diagnostic_summary_skips_nonfinite_position_rows():
    truth = pd.DataFrame(
        {
            "time_s": [0.0, np.nan, 10.0],
            "east_m": [0.0, 999.0, 10.0],
            "north_m": [0.0, 999.0, 0.0],
            "up_m": [0.0, 999.0, 0.0],
        },
        index=[4, 4, 5],
    )
    estimates = pd.DataFrame(
        {
            "time_s": [0.0, np.nan, 10.0],
            "source": ["radar", "radar", "radar"],
            "track_id": [1, 99, 1],
            "residual_norm_m": [1.0, 999.0, 2.0],
            "covariance_scale": [1.0, 999.0, 1.0],
            "east_m": [0.0, 999.0, 13.0],
            "north_m": [0.0, 999.0, 0.0],
            "up_m": [0.0, 999.0, 0.0],
        },
        index=[7, 7, 8],
    )

    summary = build_diagnostic_summary(
        estimate_frame=estimates,
        selected_radar=pd.DataFrame(),
        truth=truth,
        max_eval_time_delta_s=0.1,
        window_s=20.0,
    )

    assert len(summary["worst_time_windows"]) == 1
    window = summary["worst_time_windows"][0]
    assert window["count"] == 2
    assert math.isclose(window["rmse_3d_m"], math.sqrt(4.5))
    assert window["track_switch_count"] == 0
