from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "aggregate_stateful_lofo.py"
SPEC = importlib.util.spec_from_file_location("aggregate_stateful_lofo", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
aggregate_stateful_lofo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = aggregate_stateful_lofo
SPEC.loader.exec_module(aggregate_stateful_lofo)


def _write_text(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: object) -> None:
    _write_text(path, json.dumps(payload))


def _write_valid_flight_artifacts(artifacts_dir: Path, flight: str, *, rmse_3d_m: float = 999.0) -> None:
    root = artifacts_dir / f"stateful-lofo-{flight}"
    _write_json(
        root / "summary.json",
        {
            "flight": flight,
            "status": "ok",
            "selected_radar_rows": 5,
            "posterior_records": 10,
            "rmse_3d_m": rmse_3d_m,
            "p95_3d_m": rmse_3d_m + 1.0,
        },
    )
    _write_json(root / "radar_assoc.json", {"model": "synthetic"})
    _write_text(root / "radar_assoc_examples.csv", "feature,label\n1,1\n")
    _write_json(root / "run" / flight / "metrics.json", {"flight": flight})
    _write_json(root / "run" / flight / "diagnostic_summary.json", {"track_switch_count": 0})
    _write_text(root / "run" / flight / "diagnostics.csv", "time_s,error\n0,1\n")
    _write_text(root / "run" / flight / "selected_radar.csv", "time_s,east_m\n0,1\n")
    _write_text(root / "run" / flight / "estimates.csv", "time_s,east_m\n0,1\n")
    _write_text(root / "run" / flight / "trajectory.png", "not-empty")


def test_smoke_mode_ignores_metric_thresholds(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    _write_valid_flight_artifacts(artifacts_dir, "Opt1", rmse_3d_m=1000.0)

    result = aggregate_stateful_lofo.aggregate_lofo_artifacts(
        artifacts_dir,
        ["Opt1"],
        smoke_mode=True,
        enforce_thresholds=True,
        target_mean_rmse_3d_m=1.0,
        target_opt1_p95_3d_m=1.0,
    )

    assert result.smoke_failures == []
    assert result.threshold_failures == []
    assert not result.should_fail
    assert result.summary["mean_rmse_3d_m"] == 1000.0


def test_smoke_mode_requires_diagnostic_summary_artifact(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    _write_valid_flight_artifacts(artifacts_dir, "Opt3")
    (artifacts_dir / "stateful-lofo-Opt3" / "run" / "Opt3" / "diagnostic_summary.json").unlink()

    result = aggregate_stateful_lofo.aggregate_lofo_artifacts(
        artifacts_dir,
        ["Opt3"],
        smoke_mode=True,
        enforce_thresholds=False,
        target_mean_rmse_3d_m=95.0,
        target_opt1_p95_3d_m=200.0,
    )

    assert result.should_fail
    assert any("diagnostic_summary.json" in failure for failure in result.smoke_failures)


def test_non_smoke_mode_still_enforces_thresholds(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "artifacts"
    _write_valid_flight_artifacts(artifacts_dir, "Opt1", rmse_3d_m=1000.0)

    result = aggregate_stateful_lofo.aggregate_lofo_artifacts(
        artifacts_dir,
        ["Opt1"],
        smoke_mode=False,
        enforce_thresholds=True,
        target_mean_rmse_3d_m=95.0,
        target_opt1_p95_3d_m=200.0,
    )

    assert result.should_fail
    assert result.smoke_failures == []
    assert any("mean_rmse_3d_m" in failure for failure in result.threshold_failures)
    assert any("Opt1 p95_3d_m" in failure for failure in result.threshold_failures)
