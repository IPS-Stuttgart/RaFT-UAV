import pandas as pd
import pytest

from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_time_column_aliases


def test_time_normalizer_parses_existing_time_s_ros_stamp_dict_before_alias():
    raw = pd.DataFrame(
        {
            "time_s": [{"sec": 8, "nanosec": 250_000_000}],
            "timestamp": [99.0],
        }
    )

    rows = normalize_time_column_aliases(raw)

    assert float(rows.loc[0, "time_s"]) == pytest.approx(8.25)


def test_candidate_normalizer_keeps_canonical_time_s_ros_stamp_string():
    raw = pd.DataFrame(
        {
            "sequence_id": ["seqStamp"],
            "time_s": ['{"sec": 9, "nanosec": 500000000}'],
            "source": ["detector"],
            "x_m": [1.0],
            "y_m": [2.0],
            "z_m": [3.0],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert len(rows) == 1
    assert rows.loc[0, "sequence_id"] == "seqStamp"
    assert float(rows.loc[0, "time_s"]) == pytest.approx(9.5)
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [1.0, 2.0, 3.0]
