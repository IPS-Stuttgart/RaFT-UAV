from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.ablation_common import collect_experiment_rows, metrics_json_path


def test_collect_experiment_rows_reuses_one_shot_flights(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def run_one(config: SimpleNamespace, run_dir: Path, flight: str) -> None:
        calls.append((config.name, flight))
        metrics_path = metrics_json_path(run_dir, flight)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps({"flight": flight}), encoding="utf-8")

    rows = collect_experiment_rows(
        configs=(SimpleNamespace(name="first"), SimpleNamespace(name="second")),
        output_dir=tmp_path,
        flights=(flight for flight in ("Opt1", "Opt2")),
        skip_existing=False,
        run_one=run_one,
        make_row=lambda config, _path, metrics: {
            "config": config.name,
            "flight": metrics["flight"],
        },
    )

    assert calls == [
        ("first", "Opt1"),
        ("first", "Opt2"),
        ("second", "Opt1"),
        ("second", "Opt2"),
    ]
    assert rows == [
        {"config": "first", "flight": "Opt1"},
        {"config": "first", "flight": "Opt2"},
        {"config": "second", "flight": "Opt1"},
        {"config": "second", "flight": "Opt2"},
    ]
