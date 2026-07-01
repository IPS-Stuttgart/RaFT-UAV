import pandas as pd
import pytest

from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_time_column_aliases


def test_candidate_normalizer_accepts_whitespace_padded_headers():
    raw = pd.DataFrame(
        {
            " Sequence_ID ": ["seqA"],
            " Time_S ": [1.25],
            " Source ": ["radar"],
            " X_M ": [10.0],
            " Y_M ": [20.0],
            " Z_M ": [30.0],
        }
    )

    rows = normalize_candidate_columns(raw)

    assert rows.loc[0, "sequence_id"] == "seqA"
    assert rows.loc[0, "time_s"] == 1.25
    assert rows.loc[0, "source"] == "radar"
    assert rows.loc[0, ["x_m", "y_m", "z_m"]].tolist() == [10.0, 20.0, 30.0]


def test_time_normalizer_accepts_whitespace_padded_stamp_headers():
    raw = pd.DataFrame(
        {
            " header.stamp.sec ": [7],
            " header.stamp.nanosec ": [125_000_000],
        }
    )

    rows = normalize_time_column_aliases(raw)

    assert float(rows.loc[0, "time_s"]) == pytest.approx(7.125)


def test_time_normalizer_accepts_whitespace_padded_nested_stamp_keys():
    raw = pd.DataFrame(
        {
            "header": [
                {
                    " Stamp ": {
                        " Sec ": 8,
                        " Nanosec ": 250_000_000,
                    }
                }
            ]
        }
    )

    rows = normalize_time_column_aliases(raw)

    assert float(rows.loc[0, "time_s"]) == pytest.approx(8.25)
