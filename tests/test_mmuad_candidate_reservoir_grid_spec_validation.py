from __future__ import annotations

import pytest

from raft_uav.mmuad.candidate_reservoir_grid import _offset_config_grid
from raft_uav.mmuad.candidate_reservoir_grid import _parse_offset_specs


def test_offset_grid_deduplicates_repeated_values() -> None:
    assert _parse_offset_specs(["raw=0,0.0,1,1.0,-0"]) == [
        ("raw", (0.0, 1.0))
    ]

    configs = _offset_config_grid(["raw=0,0,1"], [])
    assert [label for label, _, _ in configs] == ["identity", "branch_raw_1"]


def test_offset_grid_rejects_duplicate_target_names() -> None:
    with pytest.raises(ValueError, match="duplicate offset grid name 'raw'"):
        _offset_config_grid(["raw=0,1", "raw=2,3"], [])


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_offset_grid_rejects_nonfinite_values(value: str) -> None:
    with pytest.raises(ValueError, match="offset values for 'raw' must be finite"):
        _parse_offset_specs([f"raw={value}"])


def test_offset_grid_rejects_empty_value_tokens() -> None:
    with pytest.raises(ValueError, match="invalid offset grid spec"):
        _parse_offset_specs(["raw=0,,1"])


def test_offset_grid_preserves_finite_negative_values() -> None:
    assert _parse_offset_specs(["raw=-1.5,0,2"]) == [
        ("raw", (-1.5, 0.0, 2.0))
    ]
