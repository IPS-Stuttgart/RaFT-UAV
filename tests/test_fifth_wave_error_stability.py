from __future__ import annotations

import numpy as np
import pandas as pd

from raft_uav.evaluation.fifth_wave_diagnostics import (
    bad_segment_table,
    block_bootstrap_interval,
    estimate_error_frame,
    oracle_replay_realistic_gap,
    paired_error_delta_frame,
    vertical_horizontal_error_summary,
)


def _position_frame(east_m: float, north_m: float, up_m: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [east_m],
            "north_m": [north_m],
            "up_m": [up_m],
        }
    )


def test_estimate_error_frame_keeps_large_representable_norms_finite() -> None:
    estimates = _position_frame(6.0e307, 8.0e307)
    truth = _position_frame(0.0, 0.0)

    with np.errstate(over="raise", invalid="raise"):
        errors = estimate_error_frame(estimates, truth)

    assert errors["error_2d_m"].tolist() == [1.0e308]
    assert errors["error_3d_m"].tolist() == [1.0e308]


def test_paired_error_delta_keeps_large_representable_norm_finite() -> None:
    method_a = _position_frame(6.0e307, 8.0e307)
    method_b = _position_frame(0.0, 0.0)
    truth = _position_frame(0.0, 0.0)

    with np.errstate(over="raise", invalid="raise"):
        delta = paired_error_delta_frame(method_a, method_b, truth)

    assert delta["error_a_m"].tolist() == [1.0e308]
    assert delta["error_b_m"].tolist() == [0.0]
    assert delta["delta_error_m"].tolist() == [1.0e308]


def test_vertical_horizontal_summary_keeps_large_horizontal_error_finite() -> None:
    estimates = _position_frame(6.0e307, 8.0e307)
    truth = _position_frame(0.0, 0.0)

    with np.errstate(over="raise", invalid="raise"):
        summary = vertical_horizontal_error_summary(estimates, truth)

    assert summary["horizontal_rmse_m"] == 1.0e308
    assert summary["horizontal_p95_m"] == 1.0e308
    assert summary["up_rmse_m"] == 0.0


def test_fifth_wave_rmse_falls_back_only_after_finite_square_overflow() -> None:
    values = np.array([8.0e307, 1.0e308])
    expected = np.sqrt(0.82) * 1.0e308

    with np.errstate(over="raise", invalid="raise"):
        interval = block_bootstrap_interval(
            values,
            metric="rmse",
            block_size=2,
            resamples=4,
            seed=0,
        )
        windows = bad_segment_table(
            np.array([0.0, 1.0]),
            values,
            window_s=2.0,
            stride_s=2.0,
            top_k=1,
        )

    assert np.isclose(interval.estimate, expected, rtol=1.0e-15)
    assert np.isfinite(interval.lower)
    assert np.isfinite(interval.upper)
    assert np.isclose(float(windows.iloc[0]["rmse_m"]), expected, rtol=1.0e-15)
    assert float(windows.iloc[0]["mae_m"]) == 9.0e307


def test_oracle_replay_summary_keeps_large_finite_statistics() -> None:
    real = np.array([8.0e307, 1.0e308])
    oracle = np.array([4.0e307, 5.0e307])

    with np.errstate(over="raise", invalid="raise"):
        summary = oracle_replay_realistic_gap(real, oracle)

    assert np.isfinite(summary["real_rmse_m"])
    assert np.isfinite(summary["oracle_replay_rmse_m"])
    assert np.isfinite(summary["association_gap_rmse_m"])
    assert np.isfinite(summary["real_p95_m"])


def test_ordinary_rmse_preserves_established_direct_result() -> None:
    values = np.array([0.1, 0.2])
    expected = float(np.sqrt(np.mean(values**2)))

    interval = block_bootstrap_interval(
        values,
        metric="rmse",
        block_size=2,
        resamples=1,
        seed=0,
    )

    assert interval.estimate == expected
