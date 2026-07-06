from __future__ import annotations

import pytest

from raft_uav.mmuad.leaderboard import leaderboard_entries_from_config


def test_leaderboard_config_rejects_negative_entry_timing(tmp_path) -> None:
    with pytest.raises(ValueError, match="max_time_delta_s"):
        leaderboard_entries_from_config(
            {
                "default_truth": "truth.csv",
                "methods": [
                    {
                        "method": "baseline",
                        "results_csv": "results.csv",
                        "max_time_delta_s": -0.1,
                    }
                ],
            },
            base_dir=tmp_path,
        )


def test_leaderboard_config_rejects_negative_default_timing(tmp_path) -> None:
    with pytest.raises(ValueError, match="default_timestamp_tolerance_s"):
        leaderboard_entries_from_config(
            {
                "default_truth": "truth.csv",
                "default_timestamp_tolerance_s": -1.0e-6,
                "methods": [{"method": "baseline", "results_csv": "results.csv"}],
            },
            base_dir=tmp_path,
        )
