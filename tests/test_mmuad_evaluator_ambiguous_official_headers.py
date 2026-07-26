from __future__ import annotations

import pandas as pd
import pytest

from raft_uav.mmuad.evaluator import validate_mmaud_results_frame


@pytest.mark.parametrize("second_sequence_header", ["Sequence", " sequence "])
def test_evaluator_rejects_ambiguous_official_sequence_headers(
    second_sequence_header: str,
) -> None:
    frame = pd.DataFrame(
        [["seq-a", "seq-b", "0.0", "[1.0, 2.0, 3.0]", "2"]],
        columns=[
            "Sequence",
            second_sequence_header,
            "Timestamp",
            "Position",
            "Classification",
        ],
    )

    with pytest.raises(
        ValueError,
        match=r"official Track 5 columns must be unique.*'sequence'",
    ):
        validate_mmaud_results_frame(frame)
