from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "find_radar_origin_candidates.py"
_SPEC = importlib.util.spec_from_file_location("find_radar_origin_candidates", _TOOL_PATH)
assert _SPEC is not None and _SPEC.loader is not None
radar_origin_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(radar_origin_tool)


def test_write_csv_preserves_explicit_empty_coordinate_schema(tmp_path: Path) -> None:
    path = tmp_path / "radar_origin_coordinate_candidates.csv"

    radar_origin_tool.write_csv(path, [], fieldnames=radar_origin_tool.COORDINATE_FIELDS)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    assert header == radar_origin_tool.COORDINATE_FIELDS
    assert rows == []


def test_write_csv_preserves_explicit_empty_match_schema(tmp_path: Path) -> None:
    path = tmp_path / "radar_origin_search_matches.csv"

    radar_origin_tool.write_csv(path, [], fieldnames=radar_origin_tool.MATCH_FIELDS)

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)

    assert header == radar_origin_tool.MATCH_FIELDS
    assert rows == []
