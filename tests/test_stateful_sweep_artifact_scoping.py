from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def load_script() -> ModuleType:
    path = Path(".github/scripts/aggregate_stateful_sweep.py")
    spec = importlib.util.spec_from_file_location("aggregate_stateful_sweep_scoping", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_summary(
    root: Path,
    artifact: str,
    *,
    flight: str,
    variant: str,
    idtp: int,
    idfp: int,
    idfn: int,
) -> None:
    path = root / artifact / "sweep_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    mot = {
        "gt": 10,
        "estimates": 10,
        "tp": 10,
        "fp": 0,
        "fn": 0,
        "idsw": 0,
        "fragmentations": 0,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
    }
    payload = {
        "flight": flight,
        "variant": variant,
        "status": "ok",
        "rmse_3d_m": 1.0,
        "p95_3d_m": 2.0,
        "selected_radar_rows": 10,
        "track_switch_count": 0,
        "selected_radar_mot": mot,
        "estimate_mot": mot,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_expected_variants_exclude_unrequested_artifacts(tmp_path: Path) -> None:
    script = load_script()
    artifacts = tmp_path / "artifacts"
    write_summary(
        artifacts,
        "requested",
        flight="Opt1",
        variant="requested",
        idtp=8,
        idfp=1,
        idfn=1,
    )
    write_summary(
        artifacts,
        "stale-perfect",
        flight="Opt1",
        variant="stale-perfect",
        idtp=10,
        idfp=0,
        idfn=0,
    )

    result = script.aggregate_sweep_artifacts(
        artifacts,
        expected_flights=["Opt1"],
        expected_variants=["requested"],
    )

    assert [row["variant"] for row in result.variants] == ["requested"]
    assert result.summary["best_variant"]["variant"] == "requested"


def test_expected_flights_exclude_unrequested_rows_from_metrics(tmp_path: Path) -> None:
    script = load_script()
    artifacts = tmp_path / "artifacts"
    write_summary(
        artifacts,
        "requested-flight",
        flight="Opt1",
        variant="candidate",
        idtp=8,
        idfp=1,
        idfn=1,
    )
    write_summary(
        artifacts,
        "stale-flight",
        flight="OldFlight",
        variant="candidate",
        idtp=0,
        idfp=100,
        idfn=100,
    )

    result = script.aggregate_sweep_artifacts(
        artifacts,
        expected_flights=["Opt1"],
        expected_variants=["candidate"],
    )

    assert result.variants[0]["selected_radar_idf1"] == pytest.approx(16.0 / 18.0)
    assert result.variants[0]["ok_flights"] == 1


def test_duplicate_requested_run_fails_instead_of_double_counting(tmp_path: Path) -> None:
    script = load_script()
    artifacts = tmp_path / "artifacts"
    for artifact, idtp in (("first", 8), ("duplicate", 10)):
        write_summary(
            artifacts,
            artifact,
            flight="Opt1",
            variant="candidate",
            idtp=idtp,
            idfp=1,
            idfn=1,
        )

    result = script.aggregate_sweep_artifacts(
        artifacts,
        expected_flights=["Opt1"],
        expected_variants=["candidate"],
    )

    assert result.should_fail is True
    assert result.summary["best_variant"] is None
    assert result.summary["failed_runs"] == ["Opt1: duplicate summaries (2)"]
    assert result.variants[0]["selected_radar_mot"]["gt"] == 0
