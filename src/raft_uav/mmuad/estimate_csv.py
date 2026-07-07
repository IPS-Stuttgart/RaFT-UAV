"""CSV readers for MMUAD estimate trajectory tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_estimate_csv(path: Path) -> pd.DataFrame:
    """Read estimate CSVs without coercing opaque identifier columns.

    Track 5 sequence identifiers can be numeric-looking strings such as ``001``.
    Read the table as text first so pandas cannot coerce those values before the
    normal schema-specific numeric conversion in downstream loaders.
    """

    rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = rows.copy()
    out.columns = [str(column).strip() for column in out.columns]
    return out
