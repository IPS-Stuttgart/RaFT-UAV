"""Package wrapper that tightens public Track 5 result Classification validation.

The legacy evaluator implementation lives in the sibling ``evaluator.py`` file.
This wrapper preserves public imports while overriding only the official-result
conversion used for submitted result rows.  Official truth-file loading remains
permissive so existing local truth archives stay readable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from raft_uav.mmuad.submission import OFFICIAL_TRACK5_CLASS_IDS

_IMPL_PATH = Path(__file__).resolve().parent.parent / "evaluator.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._evaluator_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy MMUAD evaluator from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _parse_official_result_classification_cell(value: Any) -> int:
    class_id = _IMPL.parse_official_classification_cell(value)
    if class_id not in OFFICIAL_TRACK5_CLASS_IDS:
        allowed = ", ".join(str(item) for item in sorted(OFFICIAL_TRACK5_CLASS_IDS))
        raise ValueError(
            "official MMUAD Classification values must be one of "
            f"{{{allowed}}}; got {class_id!r}"
        )
    return class_id


def _official_track5_results_to_local_frame(
    frame: pd.DataFrame,
    *,
    enforce_class_domain: bool = True,
) -> pd.DataFrame:
    lower_to_original = {str(column).lower(): column for column in frame.columns}
    sequence_col = lower_to_original["sequence"]
    timestamp_col = lower_to_original["timestamp"]
    position_col = lower_to_original["position"]
    classification_col = lower_to_original["classification"]
    sequences = [_IMPL.parse_official_sequence_cell(value) for value in frame[sequence_col]]
    timestamps = [_IMPL.parse_official_timestamp_cell(value) for value in frame[timestamp_col]]
    positions = [_IMPL.parse_official_position_cell(value) for value in frame[position_col]]
    class_parser = (
        _parse_official_result_classification_cell
        if enforce_class_domain
        else _IMPL.parse_official_classification_cell
    )
    classifications = [class_parser(value) for value in frame[classification_col]]
    xyz = pd.DataFrame(positions, columns=["x", "y", "z"], index=frame.index)
    return pd.DataFrame(
        {
            "sequence_id": sequences,
            "timestamp": timestamps,
            "x": xyz["x"],
            "y": xyz["y"],
            "z": xyz["z"],
            "uav_type": [str(value) for value in classifications],
            "score": 1.0,
        }
    )


def _official_track5_truth_to_rows(frame: pd.DataFrame) -> pd.DataFrame:
    local = _official_track5_results_to_local_frame(frame, enforce_class_domain=False)
    rows = local.rename(
        columns={
            "timestamp": "time_s",
            "x": "x_m",
            "y": "y_m",
            "z": "z_m",
            "uav_type": "class_name",
        }
    )
    return _IMPL.normalize_truth_columns(rows)


_IMPL._official_track5_results_to_local_frame = _official_track5_results_to_local_frame
_IMPL._official_track5_truth_to_rows = _official_track5_truth_to_rows

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

globals()["_parse_official_result_classification_cell"] = _parse_official_result_classification_cell
__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
