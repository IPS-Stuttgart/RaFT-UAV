"""Compatibility validation for paper-parity stage counts.

The maintained implementation lives in the sibling ``paper_parity.py`` module.
This package preserves the public import path while requiring count-like inputs
to be exact finite integers and requiring every reference count before declaring
full count parity.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib.util
from pathlib import Path
import sys
from typing import Any

from raft_uav.numeric import optional_int

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "paper_parity.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.diagnostics._paper_parity_legacy",
    _LEGACY_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load paper-parity implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)
_ORIGINAL_BUILD_BASELINE_PAPER_PARITY = _LEGACY.build_baseline_paper_parity


def _optional_int(value: Any) -> int | None:
    """Return only exact finite integer-like paper-count values."""

    return optional_int(value)


def build_baseline_paper_parity(
    *,
    stage_counts: Mapping[str, Any] | None,
    rf_rows: int,
    radar_rows: int,
    selected_radar_rows: int,
    posterior_records: int,
    accepted_by_source: Mapping[str, int],
    rejected_by_source: Mapping[str, int],
    paper_position_error_3d: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a report that only claims full parity for complete exact counts."""

    report = _ORIGINAL_BUILD_BASELINE_PAPER_PARITY(
        stage_counts=stage_counts,
        rf_rows=rf_rows,
        radar_rows=radar_rows,
        selected_radar_rows=selected_radar_rows,
        posterior_records=posterior_records,
        accepted_by_source=accepted_by_source,
        rejected_by_source=rejected_by_source,
        paper_position_error_3d=paper_position_error_3d,
    )
    count_checks = report.get("count_checks", {})
    reference_counts = report.get("reference_counts", {})
    missing = report.get("missing_reference_counts", [])
    report["all_count_matches_reference"] = bool(
        not missing
        and reference_counts
        and set(count_checks) == set(reference_counts)
        and all(
            isinstance(row, dict) and row.get("matches_reference") is True
            for row in count_checks.values()
        )
    )
    return report


_LEGACY._optional_int = _optional_int
_LEGACY.build_baseline_paper_parity = build_baseline_paper_parity

globals().update(
    {
        name: getattr(_LEGACY, name)
        for name in dir(_LEGACY)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_optional_int"] = _optional_int
globals()["build_baseline_paper_parity"] = build_baseline_paper_parity

__doc__ = _LEGACY.__doc__
__all__ = [
    name for name in dir(_LEGACY) if not (name.startswith("__") and name.endswith("__"))
]
