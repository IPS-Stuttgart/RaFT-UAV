import numpy as np
import pandas as pd
import pytest

import raft_uav.diagnostics.paper_table as paper_table_module
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


@pytest.mark.parametrize("invalid_count", [206.9, "206.5", -1, True])
def test_paper_reference_count_check_rejects_nonintegral_or_invalid_counts(
    invalid_count,
):
    table = _reference_like_table().astype(object)
    table.loc[
        table["method"] == "RF raw",
        "selected_count",
    ] = invalid_count

    check = paper_reference_count_check(table)

    assert check["passed"] is False
    failed = next(
        row
        for row in check["checks"]
        if row["method"] == "RF raw" and row["column"] == "selected_count"
    )
    assert failed["actual"] is None
    assert failed["delta"] is None
    assert failed["passed"] is False


def test_paper_reference_count_check_rejects_duplicate_reference_method_rows():
    table = _reference_like_table()
    duplicate = table.loc[table["method"] == "RF raw"].copy()
    table = pd.concat([table, duplicate], ignore_index=True)

    check = paper_reference_count_check(table)

    assert check["passed"] is False
    failed = next(
        row
        for row in check["checks"]
        if row["method"] == "RF raw" and row["column"] == "selected_count"
    )
    assert failed["actual"] is None
    assert failed["delta"] is None
    assert failed["passed"] is False


def test_paper_reference_count_check_preserves_integral_text_counts():
    table = _reference_like_table().astype(object)
    table.loc[
        table["method"] == "RF raw",
        "selected_count",
    ] = "206"

    check = paper_reference_count_check(table)

    assert check["passed"] is True


def test_paper_reference_count_check_reports_nonfinite_counts():
    for invalid_count in (float("nan"), float("inf"), float("-inf")):
        table = _reference_like_table()
        table["selected_count"] = table["selected_count"].astype(float)
        table.loc[table["method"] == "RF raw", "selected_count"] = invalid_count

        check = paper_reference_count_check(table)

        assert check["passed"] is False
        failed = next(
            row
            for row in check["checks"]
            if row["method"] == "RF raw" and row["column"] == "selected_count"
        )
        assert failed["actual"] is None
        assert failed["delta"] is None
        assert failed["passed"] is False
        assert "mismatch" in check["message"]


@pytest.mark.parametrize(
    "tolerance",
    [
        -1,
        0.5,
        True,
        np.bool_(False),
        np.nan,
        np.inf,
        1.0 + 0.0j,
        np.array([1]),
        np.ma.masked,
    ],
)
def test_paper_reference_count_check_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance must be a non-negative integer scalar"):
        paper_reference_count_check(_reference_like_table(), tolerance=tolerance)


@pytest.mark.parametrize(
    ("tolerance", "expected"),
    [
        (0, 0),
        ("2", 2),
        (np.int64(3), 3),
        (np.array(4), 4),
    ],
)
def test_paper_reference_count_check_normalizes_valid_tolerance(tolerance, expected):
    check = paper_reference_count_check(_reference_like_table(), tolerance=tolerance)

    assert check["tolerance"] == expected


def test_run_paper_table_diagnostic_rejects_invalid_tolerance_before_io(monkeypatch):
    def unexpected_run(**_kwargs):
        raise AssertionError("paper-table implementation should not run")

    monkeypatch.setattr(
        paper_table_module,
        "_ORIGINAL_RUN_PAPER_TABLE_DIAGNOSTIC",
        unexpected_run,
    )

    with pytest.raises(
        ValueError,
        match="reference_count_tolerance must be a non-negative integer scalar",
    ):
        paper_table_module.run_paper_table_diagnostic(
            dataset_root=".",
            flight_name="dummy",
            reference_count_tolerance=0.5,
        )
