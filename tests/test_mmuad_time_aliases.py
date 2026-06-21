from __future__ import annotations

import pandas as pd

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
