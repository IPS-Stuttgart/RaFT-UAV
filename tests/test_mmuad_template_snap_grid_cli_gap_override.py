from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
MODULE_PATH = SCRIPTS_DIR / "mmuad_template_snap_grid.py"
spec = importlib.util.spec_from_file_location(
    "mmuad_template_snap_grid_cli_gap_override",
    MODULE_PATH,
)
assert spec is not None and spec.loader is not None
snap_grid = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = snap_grid
spec.loader.exec_module(snap_grid)


def _captured_gap_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra_args: list[str],
) -> tuple[float | None, ...]:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        snap_grid,
        "load_official_track5_results_frame",
        lambda _path: pd.DataFrame(),
    )
    monkeypatch.setattr(
        snap_grid,
        "load_official_track5_template_file",
        lambda _path: pd.DataFrame(),
    )

    def fake_run_template_snap_grid(**kwargs: object) -> pd.DataFrame:
        captured["gaps"] = kwargs["max_interpolation_gaps_s"]
        return pd.DataFrame()

    monkeypatch.setattr(snap_grid, "run_template_snap_grid", fake_run_template_snap_grid)
    rc = snap_grid.main(
        [
            "--results",
            str(tmp_path / "results.csv"),
            "--template",
            str(tmp_path / "template.csv"),
            "--output-dir",
            str(tmp_path / "out"),
            *extra_args,
        ]
    )

    assert rc == 0
    return tuple(captured["gaps"])


def test_explicit_gap_values_replace_the_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gaps = _captured_gap_values(
        monkeypatch,
        tmp_path,
        ["--max-interpolation-gap-s", "4.0"],
    )

    assert gaps == (4.0,)


def test_omitted_gap_values_keep_the_unbounded_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    gaps = _captured_gap_values(monkeypatch, tmp_path, [])

    assert gaps == (None,)
