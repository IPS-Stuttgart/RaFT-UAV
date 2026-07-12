"""Tracking/evaluation helpers for LOFO calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from raft_uav.evaluation.metrics import position_errors_m, summarize_errors


_ESTIMATE_COLUMNS = (
    "time_s",
    "source",
    "east_m",
    "north_m",
    "up_m",
    "v_east_mps",
    "v_north_mps",
    "v_up_mps",
)


def tracking_metrics(
    *,
    flight_name: str,
    truth: pd.DataFrame,
    rf: pd.DataFrame,
    radar: pd.DataFrame,
    selected: pd.DataFrame,
    estimates: pd.DataFrame,
) -> dict[str, Any]:
    """Compute core tracking metrics for the corrected held-out flight."""

    truth_times = truth["time_s"].to_numpy(dtype=float)
    truth_positions = truth[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    estimate_times = estimates["time_s"].to_numpy(dtype=float)
    estimate_positions = estimates[["east_m", "north_m", "up_m"]].to_numpy(dtype=float)
    return {
        "flight": flight_name,
        "method": "tracklet-viterbi-lofo-time-offset-bias",
        "rf_rows": int(len(rf)),
        "radar_rows": int(len(radar)),
        "selected_radar_rows": int(len(selected)),
        "posterior_records": int(len(estimates)),
        "position_error_2d": summarize_errors(
            position_errors_m(
                estimate_times,
                estimate_positions,
                truth_times,
                truth_positions,
                dimensions=2,
            )
        ),
        "position_error_3d": summarize_errors(
            position_errors_m(
                estimate_times,
                estimate_positions,
                truth_times,
                truth_positions,
                dimensions=3,
            )
        ),
    }


def records_to_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Convert tracker records into the estimates.csv schema used by reports."""

    rows = []
    for record in records:
        state = np.asarray(record["state"], dtype=float).reshape(6)
        rows.append(
            {
                "time_s": float(record["time_s"]),
                "source": str(record["source"]),
                "east_m": state[0],
                "north_m": state[1],
                "up_m": state[2],
                "v_east_mps": state[3],
                "v_north_mps": state[4],
                "v_up_mps": state[5],
            }
        )
    return (
        pd.DataFrame.from_records(rows, columns=_ESTIMATE_COLUMNS)
        .sort_values("time_s")
        .reset_index(drop=True)
    )
