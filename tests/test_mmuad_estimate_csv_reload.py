from __future__ import annotations

import importlib
from io import StringIO

import pandas as pd

import raft_uav.mmuad.estimate_csv as estimate_csv


def test_estimate_csv_guard_survives_module_reload() -> None:
    original_reader = estimate_csv._ORIGINAL_PANDAS_READ_CSV

    reloaded = importlib.reload(estimate_csv)
    rows = pd.read_csv(StringIO("value\n1\n"))

    assert rows["value"].tolist() == [1]
    assert reloaded._ORIGINAL_PANDAS_READ_CSV is original_reader
