from __future__ import annotations

import numpy as np
import pytest

from raft_uav.mmuad.leaderboard import leaderboard_entries_from_config


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
@pytest.mark.parametrize(
    ("scope", "field"),
    [
        ("default", "default_max_time_delta_s"),
        ("default", "default_timestamp_tolerance_s"),
        ("entry", "max_time_delta_s"),
        ("entry", "timestamp_tolerance_s"),
    ],
)
def test_leaderboard_config_rejects_boolean_timing_values(
    tmp_path,
    value,
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
