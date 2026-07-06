from __future__ import annotations

import pandas as pd

from raft_uav.mmuad.evaluate import match_submission_to_truth


def test_numeric_id_placeholder() -> None:
    values = [1.0, 2.0]
    assert values == [1.0, 2.0]
