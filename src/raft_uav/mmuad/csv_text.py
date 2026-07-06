"""CSV loading helpers that keep opaque sequence identifiers textual."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

SEQUENCE_ID_COLUMNS = ("Sequence", "sequence", "sequence_id", "seq", "scene", "scene_id")


def read_csv_preserving_sequence_ids(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV without converting sequence identifiers such as 001 to 1."""

    dtype = dict(kwargs.pop("dtype", {}) or {})
    for column in SEQUENCE_ID_COLUMNS:
        dtype.setdefault(column, "string")
    return pd.read_csv(path, dtype=dtype, **kwargs)
