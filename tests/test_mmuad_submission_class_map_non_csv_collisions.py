from __future__ import annotations

import json
from pathlib import Path

import pytest

from raft_uav.mmuad.submission import load_sequence_class_map


def _write_class_map(path: Path, *, second_uav_type: int) -> None:
    if path.suffix == ".json":
        path.write_text(
            json.dumps(
                {
                    "class_map": {
                        "001": 2,
                        " 001 ": second_uav_type,
                    }
                }
            ),
            encoding="utf-8",
        )
        return

    path.write_text(
        "sequences:\n"
        '  - sequence_id: "001"\n'
        "    uav_type: 2\n"
        '  - sequence_id: " 001 "\n'
        f"    uav_type: {second_uav_type}\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_non_csv_class_map_rejects_conflicting_normalized_sequence_ids(
    tmp_path: Path,
    suffix: str,
) -> None:
    class_map_path = tmp_path / f"class_map{suffix}"
    _write_class_map(class_map_path, second_uav_type=3)

    with pytest.raises(
        ValueError,
        match=(
            "JSON/YAML class map assigns conflicting UAV types to "
            "normalized sequence '001'"
        ),
    ):
        load_sequence_class_map(class_map_path)


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_non_csv_class_map_allows_matching_normalized_sequence_ids(
    tmp_path: Path,
    suffix: str,
) -> None:
    class_map_path = tmp_path / f"class_map{suffix}"
    _write_class_map(class_map_path, second_uav_type=2)

    assert load_sequence_class_map(class_map_path) == {"001": "2"}
