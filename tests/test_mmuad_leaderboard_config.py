from __future__ import annotations

import pytest

from raft_uav.mmuad.leaderboard import leaderboard_entries_from_config, load_leaderboard_config


def test_leaderboard_csv_config_ignores_blank_optional_class_map_and_note(tmp_path):
    config = tmp_path / "leaderboard.csv"
    config.write_text(
        "method,results_csv,truth_csv,class_map_csv,source_note\n"
        "baseline,results.zip,truth.csv,,\n",
        encoding="utf-8",
    )

    entries = load_leaderboard_config(config)

    assert len(entries) == 1
    entry = entries[0]
    assert entry.method == "baseline"
    assert entry.results_path == tmp_path / "results.zip"
    assert entry.truth_path == tmp_path / "truth.csv"
    assert entry.class_map_path is None
    assert entry.source_note == ""


def test_leaderboard_config_uses_later_alias_when_earlier_alias_is_nan(tmp_path):
    entries = leaderboard_entries_from_config(
        {
            "methods": [
                {
                    "method": "baseline",
                    "results": float("nan"),
                    "results_csv": "results.csv",
                    "truth": float("nan"),
                    "truth_csv": "truth.csv",
                    "class_map_csv": float("nan"),
                    "metric_protocol": float("nan"),
                    "source_note": float("nan"),
                    "max_time_delta_s": float("nan"),
                    "timestamp_tolerance_s": float("nan"),
                }
            ],
            "default_metric_protocol": "public-track5",
            "default_max_time_delta_s": 0.5,
            "default_timestamp_tolerance_s": 1.0e-6,
        },
        base_dir=tmp_path,
    )

    entry = entries[0]
    assert entry.results_path == tmp_path / "results.csv"
    assert entry.truth_path == tmp_path / "truth.csv"
    assert entry.class_map_path is None
    assert entry.metric_protocol == "public-track5"
    assert entry.source_note == ""
    assert entry.max_time_delta_s == 0.5
    assert entry.timestamp_tolerance_s == 1.0e-6


def test_leaderboard_config_rejects_missing_results_aliases():
    with pytest.raises(ValueError, match="missing results/results_csv"):
        leaderboard_entries_from_config(
            {"methods": [{"method": "broken", "results_csv": float("nan"), "truth_csv": "truth.csv"}]}
        )
