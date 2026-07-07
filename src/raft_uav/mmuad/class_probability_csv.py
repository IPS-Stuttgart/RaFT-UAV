"""CSV readers for MMUAD class probability tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_class_probability_csv(path: Path) -> pd.DataFrame:
    """Read classifier CSV output while preserving sequence ids as text."""

    try:
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    except TypeError:
        rows = pd.read_csv(path)
    out = rows.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out
