from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from raft_uav.multi_uav_lts.fixed_population_cv import run_fixed_population_cv
from raft_uav.multi_uav_lts.fixed_population_grid import run_fixed_population_grid


@pytest.mark.parametrize(
    "runner",
    [run_fixed_population_grid, run_fixed_population_cv],
    ids=["grid", "cross-validation"],
)
@pytest.mark.parametrize(
    ("source_name", "message"),
    [
        ("prediction_path", "prediction path"),
        ("truth_dir", "truth directory"),
        ("first_frame_label_dir", "first-frame label directory"),
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["target", "nested"])
def test_best_prediction_copy_guard_preserves_input_sources(
    tmp_path: Path,
    runner: Callable[..., Any],
    source_name: str,
    message: str,
    nested: bool,
) -> None:
    output_dir = tmp_path / "output"
    target = output_dir / "best_predictions"
    source = target / "source" if nested else target
    source.mkdir(parents=True)
    marker = source / "input.txt"
    marker.write_text("do not delete\n", encoding="utf-8")

    arguments = {
        "prediction_path": tmp_path / "predictions",
        "truth_dir": tmp_path / "truth",
        "first_frame_label_dir": tmp_path / "labels",
        "output_dir": output_dir,
    }
    arguments[source_name] = source

    with pytest.raises(ValueError, match=message):
        runner(**arguments)

    assert marker.read_text(encoding="utf-8") == "do not delete\n"
    assert not (output_dir / "configs").exists()
