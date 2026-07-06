"""CSV readers for MMUAD class-probability tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_class_probability_csv(path: Path) -> pd.DataFrame:
    """Read classifier CSV output without coercing opaque sequence identifiers."""

    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except TypeError:
        return pd.read_csv(path)
