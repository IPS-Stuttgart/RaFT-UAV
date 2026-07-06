from __future__ import annotations

import pandas as pd


def test_numeric_id_placeholder() -> None:
    values = [1.0, 2.0]
    assert values == [1.0, 2.0]
