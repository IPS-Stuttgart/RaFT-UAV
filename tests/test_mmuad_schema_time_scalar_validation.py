import numpy as np
import pandas as pd
import pytest

from raft_uav.mmuad.schema import (
    normalize_candidate_columns,
    normalize_time_column_aliases,
)


def test_candidate_normalizer_drops_boolean_and_complex_timestamps():
    raw = pd.DataFrame(
        {
            "time_s": [True, np.bool_(False), 1.0 + 2.0j, np.complex128(3.0), "4.5"],
            "x_m": [1.0] * 5,
            "y_m": [2.0] * 5,
            "z_m": [3.0] * 5,
        }
    )

    rows = normalize_candidate_columns(raw)

    assert len(rows) == 1
    assert float(rows.loc[0, "time_s"]) == pytest.approx(4.5)


def test_time_normalizer_falls_back_past_malformed_stamp_pair_components():
    raw = pd.DataFrame(
        {
            "header.stamp.sec": [True, 1],
            "header.stamp.nanosec": [0, True],
            "timestamp": [4.5, 5.5],
        }
    )

    rows = normalize_time_column_aliases(raw)

    assert rows["time_s"].tolist() == pytest.approx([4.5, 5.5])


def test_time_normalizer_rejects_boolean_ros_stamp_components():
    raw = pd.DataFrame(
        {
            "stamp": [
                {"sec": True},
                {"sec": 1, "nanosec": True},
                {"sec": 2, "nanosec": 500_000_000},
            ]
        }
    )

    rows = normalize_time_column_aliases(raw)

    assert rows["time_s"].iloc[:2].isna().all()
    assert float(rows.loc[2, "time_s"]) == pytest.approx(2.5)


def test_time_normalizer_handles_cyclic_stamp_mapping_as_missing():
    cyclic_stamp = {}
    cyclic_stamp["stamp"] = cyclic_stamp

    rows = normalize_time_column_aliases(pd.DataFrame({"timestamp": [cyclic_stamp]}))

    assert pd.isna(rows.loc[0, "time_s"])
