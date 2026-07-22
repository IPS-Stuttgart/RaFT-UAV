from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import raft_uav.mmuad.coordinate_alignment_audit as coordinate_audit


@pytest.mark.parametrize(
    "gate",
    [
        -0.1,
        np.nan,
        np.inf,
        -np.inf,
        True,
        np.bool_(False),
        1.0 + 0.0j,
        np.asarray([0.5]),
        np.ma.masked,
    ],
)
def test_coordinate_alignment_audit_rejects_invalid_time_gates(gate: object) -> None:
    with pytest.raises(ValueError, match="max_time_delta_s"):
        coordinate_audit.build_coordinate_alignment_audit(
            Path("unused-sequences"),
            Path("unused-truth.csv"),
            max_time_delta_s=gate,
        )


def test_coordinate_alignment_audit_normalizes_zero_dimensional_time_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build(*args: object, **kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame({"status": ["ok"]})

    monkeypatch.setattr(
        coordinate_audit,
        "_ORIGINAL_BUILD_COORDINATE_ALIGNMENT_AUDIT",
        fake_build,
    )

    result = coordinate_audit.build_coordinate_alignment_audit(
        Path("unused-sequences"),
        Path("unused-truth.csv"),
        max_time_delta_s=np.asarray(0.0),
    )

    assert result["status"].tolist() == ["ok"]
    assert captured["max_time_delta_s"] == 0.0
    assert isinstance(captured["max_time_delta_s"], float)


def test_coordinate_alignment_cli_parser_rejects_nonfinite_gates() -> None:
    assert coordinate_audit._parse_max_time_delta("unbounded") is None
    assert coordinate_audit._parse_max_time_delta("0.25") == 0.25

    with pytest.raises(ValueError, match="max_time_delta_s"):
        coordinate_audit._parse_max_time_delta("nan")
