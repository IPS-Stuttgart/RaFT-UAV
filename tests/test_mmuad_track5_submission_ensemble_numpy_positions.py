from __future__ import annotations

from pathlib import Path

import pandas as pd

from raft_uav.mmuad import run
from raft_uav.mmuad.track5_submission_ensemble import load_track5_submission


def test_track5_submission_ensemble_accepts_numpy_array_position_repr(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "Sequence": ["seq0001"],
            "Timestamp": [0.0],
            "Position": ["array([1.5, 2.5, 3.5])"],
            "Classification": [2],
        }
    ).to_csv(path, index=False)

    loaded = load_track5_submission(path)

    assert loaded.loc[0, ["state_x_m", "state_y_m", "state_z_m"]].tolist() == [
        1.5,
        2.5,
        3.5,
    ]


def test_mmuad_run_option_arity_uses_exact_flag_names() -> None:
    assert run._option_consumes_next("--inspect-layout-only") is False
    assert run._option_consumes_next("--future-value-only") is True
    assert run._option_consumes_next("--future-value-from-candidates") is True


def test_mmuad_run_known_flag_only_option_does_not_consume_sequence_root(monkeypatch) -> None:
    forwarded: list[str] = []

    def fake_track_main(argv: list[str] | None = None) -> int:
        forwarded.extend(argv or [])
        return 0

    monkeypatch.setattr(run, "track_main", fake_track_main)

    assert run.main(["--inspect-layout-only", "data/mmuad", "--output-dir", "outputs/mmuad"]) == 0
    assert forwarded == [
        "--sequence-root",
        "data/mmuad",
        "--inspect-layout-only",
        "--output-dir",
        "outputs/mmuad",
    ]
