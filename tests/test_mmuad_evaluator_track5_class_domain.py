from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import validate_mmaud_results_frame


def _official_result_frame(classification: int | str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Sequence": ["seq1"],
            "Timestamp": [0.0],
            "Position": ["(1.0,2.0,3.0)"],
            "Classification": [classification],
        }
    )


def test_official_track5_results_evaluator_accepts_declared_class_domain() -> None:
    rows = validate_mmaud_results_frame(_official_result_frame(3))

    assert rows.loc[0, "sequence_id"] == "seq1"
    assert rows.loc[0, "uav_type"] == "3"


def test_official_track5_results_evaluator_rejects_out_of_domain_classification() -> None:
    with pytest.raises(ValueError, match=r"must be one of \{0, 1, 2, 3\}"):
        validate_mmaud_results_frame(_official_result_frame(4))
