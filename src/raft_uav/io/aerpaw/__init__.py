from __future__ import annotations

import importlib.util as _importlib_util
import json as _json
from pathlib import Path as _Path
from typing import Any as _Any

import pandas as _pd

_LEGACY_PATH = _Path(__file__).resolve().parent.parent / "aerpaw.py"
_SPEC = _importlib_util.spec_from_file_location("_raft_uav_io_aerpaw_legacy", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_LEGACY_PATH}")
_legacy = _importlib_util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_legacy)


def read_radar_tracks_json(path: _Path) -> _pd.DataFrame:
    """Read Fortem newline-delimited radar JSON logs into one row per track."""

    records: list[dict[str, _Any]] = []
    saw_non_object_payload = False
    logical_frame_index = 0
    with _Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            frame_index = logical_frame_index
            logical_frame_index += 1
            try:
                payload = _json.loads(line)
            except _json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid radar JSON on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                saw_non_object_payload = True
                continue

            params = payload.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                params = {}

            track_data = _legacy._track_data_from_payload(payload, params)
            if not isinstance(track_data, list):
                continue
            for track_index, track in enumerate(track_data):
                if not isinstance(track, dict):
                    continue
                records.append(
                    _legacy._flatten_track(frame_index, track_index, track, params)
                )
    if saw_non_object_payload and not records:
        raise ValueError("radar JSON must contain a JSON object")
    return _pd.DataFrame.from_records(records)


_legacy.read_radar_tracks_json = read_radar_tracks_json

for _name, _value in vars(_legacy).items():
    if not _name.startswith("_"):
        globals()[_name] = _value
globals()["read_radar_tracks_json"] = read_radar_tracks_json
__all__ = sorted(_name for _name in globals() if not _name.startswith("_"))
