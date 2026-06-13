"""Backward-compatible import shim for Track 5 submission validation.

The invalid sequence/timestamp validity guard now lives in
``raft_uav.mmuad.submission`` itself.  This module remains importable for older
code paths that referenced the temporary guard module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.submission import validate_official_track5_submission as _validate


def validate_official_track5_submission(
    path: Path,
    *,
    template: pd.DataFrame | None = None,
    timestamp_tolerance_s: float = 1.0e-6,
    require_zip: bool = True,
) -> Any:
    """Validate a Track 5 submission using the canonical implementation."""

    return _validate(
        path,
        template=template,
        timestamp_tolerance_s=timestamp_tolerance_s,
        require_zip=require_zip,
    )
