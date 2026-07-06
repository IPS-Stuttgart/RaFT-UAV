from __future__ import annotations

import pandas as pd

from raft_uav.mmuad import run
from raft_uav.mmuad.schema import normalize_time_column_aliases


def test_stamp_dict_unit_timestamp_aliases_are_scaled_to_seconds() -> None:
    frame = pd.DataFrame(
        {
            "stamp": [
                {"timestamp_us": 1_250_000},
                {"time_ms": 2500},
                {"stamp_ns": 3_000_000_000},
            ]
        }
    )

    normalized = normalize_time_column_aliases(frame)

    assert normalized["time_s"].tolist() == [1.25, 2.5, 3.0]


def test_stringified_stamp_dict_timestamp_aliases_are_scaled_to_seconds() -> None:
    frame = pd.DataFrame(
        {
            "stamp": [
                "{'sec': 8, 'nanosec': 250000000}",
                '{"timestamp_us": 1250000}',
                "{'stamp': {'time_ms': 2500}}",
            ]
        }
    )

    normalized = normalize_time_column_aliases(frame)

    assert normalized["time_s"].tolist() == [8.25, 1.25, 2.5]


def test_header_stamp_dict_unit_timestamp_aliases_are_scaled_to_seconds() -> None:
    frame = pd.DataFrame(
        {
            "header": [
                {"stamp": {"timestamp_us": 1_500_000}},
                {"stamp": {"time_ms": 2500}},
                {"stamp": {"stamp_ns": 3_000_000_000}},
            ]
        }
    )

    normalized = normalize_time_column_aliases(frame)

    assert normalized["time_s"].tolist() == [1.5, 2.5, 3.0]


def test_second_fraction_pair_aliases_keep_valid_integer_seconds() -> None:
    frame = pd.DataFrame(
        {
            "timestamp_sec": [10, 11, None],
            "timestamp_nanosec": [pd.NA, pd.NA, 500_000_000],
        }
    )

    normalized = normalize_time_column_aliases(frame)

    assert normalized["time_s"].iloc[:2].tolist() == [10.0, 11.0]
    assert pd.isna(normalized["time_s"].iloc[2])


def test_stamp_dict_aliases_keep_valid_integer_seconds() -> None:
    frame = pd.DataFrame(
        {
            "stamp": [
                {"sec": 12, "nanosec": pd.NA},
                {"stamp": {"secs": 13, "nsecs": pd.NA}},
            ]
        }
    )

    normalized = normalize_time_column_aliases(frame)

    assert normalized["time_s"].tolist() == [12.0, 13.0]


def test_existing_time_s_is_filled_rowwise_from_timestamp_aliases() -> None:
    frame = pd.DataFrame(
        {
            "time_s": [1.0, None, "bad", 4.0],
            "timestamp_us": [100_000, 2_000_000, 3_000_000, 400_000],
        }
    )

    normalized = normalize_time_column_aliases(frame)

    assert normalized["time_s"].tolist() == [1.0, 2.0, 3.0, 4.0]


def test_run_option_classifier_defaults_unlisted_long_options_to_value_taking() -> None:
    assert run._option_consumes_next("--new-value-option")
    assert not run._option_consumes_next("--new-value-option=configured-value")
    assert not run._option_consumes_next("--inspect-layout-only")
