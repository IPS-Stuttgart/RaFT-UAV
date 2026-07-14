from __future__ import annotations

import pandas as pd
import pytest

from raft_uav import lofo_time_offset_cli


def test_main_rejects_duplicate_resolved_flights(monkeypatch, tmp_path) -> None:
    flight = lofo_time_offset_cli._LoadedFlight(
        name="Opt1",
        truth=pd.DataFrame(),
        rf=pd.DataFrame(),
        radar=pd.DataFrame(),
    )

    monkeypatch.setattr(lofo_time_offset_cli, "_load_flight", lambda _root, _name: flight)

    output_dir = tmp_path / "output"
    with pytest.raises(
        ValueError,
        match=r"LOFO calibration needs distinct flights; duplicate resolved flight\(s\): Opt1",
    ):
        lofo_time_offset_cli.main(
            [
                str(tmp_path),
                "--flight",
                "Opt1",
                "--flight",
                "Opt1",
                "--output-dir",
                str(output_dir),
                "--skip-tracking",
            ]
        )

    assert not output_dir.exists()
