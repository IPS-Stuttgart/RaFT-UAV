from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

from scripts import run_determinism_check


def _write_estimates(run_dir: Path) -> None:
    run_dir.mkdir()
    pd.DataFrame(
        {
            "time_s": [0.0],
            "east_m": [1.0],
            "north_m": [2.0],
            "up_m": [3.0],
        }
    ).to_csv(run_dir / "estimates.csv", index=False)


@pytest.mark.parametrize("selected_side", ["a", "b"])
def test_fail_on_difference_for_one_sided_selected_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected_side: str,
) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    _write_estimates(run_a)
    _write_estimates(run_b)
    selected_run = run_a if selected_side == "a" else run_b
    pd.DataFrame({"time_s": [0.0], "track_id": [7]}).to_csv(
        selected_run / "selected_radar.csv",
        index=False,
    )
    output_json = tmp_path / "summary.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_determinism_check.py",
            str(run_a),
            str(run_b),
            "--output-json",
            str(output_json),
            "--fail-on-difference",
        ],
    )

    assert run_determinism_check.main() == 1
    summary = json.loads(output_json.read_text(encoding="utf-8"))
    assert summary["selected_artifact_present_a"] is (selected_side == "a")
    assert summary["selected_artifact_present_b"] is (selected_side == "b")
    assert summary["selected_artifact_presence_equal"] is False
    assert summary["selected_rows_equal"] is False
