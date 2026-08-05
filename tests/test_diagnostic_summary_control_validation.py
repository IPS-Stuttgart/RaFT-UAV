from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from raft_uav.evaluation.diagnostics import build_diagnostic_summary


def _build_summary(
    *,
    top_n: object = 20,
    window_s: object = 30.0,
    max_eval_time_delta_s: object | None = None,
) -> dict[str, object]:
    return build_diagnostic_summary(
        estimate_frame=pd.DataFrame(),
        selected_radar=pd.DataFrame(),
        truth=pd.DataFrame(),
        max_eval_time_delta_s=max_eval_time_delta_s,
        top_n=top_n,
        window_s=window_s,
    )


@pytest.mark.parametrize(
    "top_n",
    [
        True,
        np.bool_(True),
        1.5,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([1]),
    ],
)
def test_diagnostic_summary_rejects_invalid_top_n(top_n: object) -> None:
    with pytest.raises(ValueError, match="top_n"):
        _build_summary(top_n=top_n)


@pytest.mark.parametrize(
    "window_s",
    [
        True,
        np.bool_(True),
        0.0,
        -1.0,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([1.0]),
    ],
)
def test_diagnostic_summary_rejects_invalid_window_s(window_s: object) -> None:
    with pytest.raises(ValueError, match="window_s"):
        _build_summary(window_s=window_s)


@pytest.mark.parametrize(
    "max_eval_time_delta_s",
    [
        True,
        np.bool_(False),
        -1.0,
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([1.0]),
    ],
)
def test_diagnostic_summary_rejects_invalid_time_gate(
    max_eval_time_delta_s: object,
) -> None:
    with pytest.raises(ValueError, match="max_eval_time_delta_s"):
        _build_summary(max_eval_time_delta_s=max_eval_time_delta_s)


def test_diagnostic_summary_accepts_zero_dimensional_real_controls() -> None:
    summary = _build_summary(
        top_n=np.array(2),
        window_s=np.array(5.0),
        max_eval_time_delta_s=np.array(0.0),
    )

    assert summary["top_n"] == 2
    assert summary["window_s"] == 5.0
