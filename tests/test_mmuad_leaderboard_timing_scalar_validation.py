from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.leaderboard import leaderboard_entries_from_config


@pytest.mark.parametrize(
    "value",
    [
        np.array(True),
        np.array(False),
        np.array([0.25]),
        np.array([[0.25]]),
        0.25 + 0.0j,
        np.ma.masked,
    ],
)
@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("default", "default_max_time_delta_s"),
        ("default", "default_timestamp_tolerance_s"),
        ("entry", "max_time_delta_s"),
        ("entry", "timestamp_tolerance_s"),
    ],
)
def test_leaderboard_config_rejects_non_real_scalar_timing_values(
    tmp_path,
    value: object,
    scope: str,
    field: str,
) -> None:
    payload = {
        "default_truth": "truth.csv",
        "methods": [{"method": "baseline", "results_csv": "results.csv"}],
    }
    target = payload if scope == "default" else payload["methods"][0]
    target[field] = value

    with pytest.raises(ValueError, match=field):
        leaderboard_entries_from_config(payload, base_dir=tmp_path)


@pytest.mark.parametrize("value", [np.float64(0.25), np.array(0.25), "0.25"])
@pytest.mark.parametrize(
    ("scope", "field", "attribute"),
    [
        ("default", "default_max_time_delta_s", "max_time_delta_s"),
        (
            "default",
            "default_timestamp_tolerance_s",
            "timestamp_tolerance_s",
        ),
        ("entry", "max_time_delta_s", "max_time_delta_s"),
        ("entry", "timestamp_tolerance_s", "timestamp_tolerance_s"),
    ],
)
def test_leaderboard_config_accepts_real_scalar_like_timing_values(
    tmp_path,
    value: object,
    scope: str,
    field: str,
    attribute: str,
) -> None:
    payload = {
        "default_truth": "truth.csv",
        "methods": [{"method": "baseline", "results_csv": "results.csv"}],
    }
    target = payload if scope == "default" else payload["methods"][0]
    target[field] = value

    entry = leaderboard_entries_from_config(payload, base_dir=tmp_path)[0]

    assert getattr(entry, attribute) == pytest.approx(0.25)
