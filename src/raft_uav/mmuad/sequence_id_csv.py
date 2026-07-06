"""CSV helpers for preserving MMUAD sequence identifiers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_sequence_id_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"sequence_id": "string", "Sequence": "string"})
