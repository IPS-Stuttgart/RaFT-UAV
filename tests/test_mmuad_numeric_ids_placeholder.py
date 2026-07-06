from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluate import match_submission_to_truth


def test_numeric_id_placeholder() -> None:
    frame = pd.DataFrame.from_records([{"id": 1.0}, {"id": 2.0}])
    assert frame["id"].tolist() == [1.0, 2.0]
