"""Compatibility wrapper for robust Fortem JSONL radar loading."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "aerpaw.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.io._aerpaw_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load AERPAW implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _track_data_from_payload(
    payload: dict[str, Any],
    params: dict[str, Any],
) -> Any:
    """Return the first list-valued Fortem track-data representation.

    Some JSON-RPC logs include a null or malformed top-level ``trackData``
    placeholder while storing the actual list in ``params.trackData``. A valid
    top-level list keeps precedence; otherwise a valid nested list is used.
    """

    top_level = payload.get("trackData")
    if isinstance(top_level, list):
        return top_level

    nested = params.get("trackData")
    if isinstance(nested, list):
        return nested

    if top_level is not None:
        return top_level
    return [] if nested is None else nested


def read_radar_tracks_json(path: Path) -> pd.DataFrame:
    """Read Fortem JSONL while indexing non-empty records rather than physical lines."""

    records: list[dict[str, Any]] = []
    saw_non_object_payload = False
    frame_index = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid radar JSON on line {line_number}: {exc.msg}"
                ) from exc

            current_frame_index = frame_index
            frame_index += 1
            if not isinstance(payload, dict):
                saw_non_object_payload = True
                continue

            params = payload.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                params = {}

            track_data = _track_data_from_payload(payload, params)
            if not isinstance(track_data, list):
                continue
            for track_index, track in enumerate(track_data):
                if not isinstance(track, dict):
                    continue
                records.append(
                    _IMPL._flatten_track(current_frame_index, track_index, track, params)
                )
    if saw_non_object_payload and not records:
        raise ValueError("radar JSON must contain a JSON object")
    return pd.DataFrame.from_records(records)


_IMPL._track_data_from_payload = _track_data_from_payload
_IMPL.read_radar_tracks_json = read_radar_tracks_json

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_track_data_from_payload"] = _track_data_from_payload
globals()["read_radar_tracks_json"] = read_radar_tracks_json

__doc__ = _IMPL.__doc__
__all__ = [name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))]
