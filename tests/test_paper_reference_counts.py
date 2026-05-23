import pandas as pd

from raft_uav.diagnostics.paper_table import paper_reference_count_check


def _reference_like_table():
    return pd.DataFrame(
        [
            {"method": "RF raw", "selected_count": 206},
            {"method": "radar-longest-continuous-track-range-gated", "selected_count": 2403},
            {
                "method": "fusion-paper-compatible",
                "selected_count": 2655,
                "accepted_measurements": 2528,
                "coasted_measurements": 127,
            },
        ]
    )


def test_paper_reference_count_check_passes_for_reference_counts():
    check = paper_reference_count_check(_reference_like_table())

    assert check["passed"] is True


def test_paper_reference_count_check_flags_mismatch():
    table = _reference_like_table()
    table.loc[table["method"] == "fusion-paper-compatible", "coasted_measurements"] = 126

    check = paper_reference_count_check(table)

    assert check["passed"] is False
    assert "mismatch" in check["message"]
