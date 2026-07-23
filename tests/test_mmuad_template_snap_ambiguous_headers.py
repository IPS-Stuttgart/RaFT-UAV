from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.template_snap_core import snap_official_results_to_template


def _valid_results() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq001"],
            "Timestamp": [1.0],
            "Position": ["(1,2,3)"],
            "Classification": [1],
        }
    )


def _valid_template() -> pd.DataFrame:
    return pd.DataFrame({"Sequence": ["seq001"], "Timestamp": [1.0]})


def test_template_snap_rejects_ambiguous_result_headers() -> None:
    results = pd.DataFrame(
        [["seq001", "seq999", 1.0, "(1,2,3)", 1]],
        columns=[
            "Sequence",
            " sequence ",
            "Timestamp",
            "Position",
            "Classification",
        ],
    )

    with pytest.raises(
        ValueError,
        match="official Track 5 results contains ambiguous columns",
    ):
        snap_official_results_to_template(results, _valid_template())


def test_template_snap_rejects_ambiguous_template_headers() -> None:
    template = pd.DataFrame(
        [["seq001", "seq999", 1.0]],
        columns=["Sequence", " SEQUENCE ", "Timestamp"],
    )

    with pytest.raises(
        ValueError,
        match="Track 5 template contains ambiguous columns",
    ):
        snap_official_results_to_template(_valid_results(), template)


def test_template_snap_preserves_single_padded_headers() -> None:
    results = pd.DataFrame(
        [["seq001", 1.0, "(1,2,3)", 1]],
        columns=[" sequence ", " timestamp ", " position ", " classification "],
    )
    template = pd.DataFrame(
        [["seq001", 1.0]],
        columns=[" sequence ", " timestamp "],
    )

    snapped, _ = snap_official_results_to_template(results, template)

    assert snapped.loc[0, "Position"] == "(1,2,3)"
    assert int(snapped.loc[0, "Classification"]) == 1
