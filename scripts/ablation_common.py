"""Shared helpers for RaFT-UAV ablation and metric-collection scripts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TypeVar

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FLIGHTS = ["Opt1", "Opt2", "Opt3"]

ConfigT = TypeVar("ConfigT")


def add_experiment_io_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path,
    default_summary_output: Path,
    default_flights: Sequence[str] = DEFAULT_FLIGHTS,
) -> None:
    """Add dataset/output/flight arguments shared by ablation runners."""

    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--summary-output", type=Path, default=default_summary_output)
    parser.add_argument("--flights", nargs="*", default=list(default_flights))


def add_soft_update_arguments(parser: argparse.ArgumentParser) -> None:
    """Add NIS-inflation arguments shared by soft-update ablations."""

    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)


def add_fixed_lag_argument(parser: argparse.ArgumentParser) -> None:
    """Add the fixed-lag smoothing horizon argument."""

    parser.add_argument("--fixed-lag-s", type=float, default=20.0)


def collect_experiment_rows(
    *,
    configs: Iterable[ConfigT],
    output_dir: Path,
    flights: Iterable[str],
    skip_existing: bool,
    run_one: Callable[[ConfigT, Path, str], None],
    make_row: Callable[[ConfigT, Path, dict[str, Any]], dict[str, object]],
) -> list[dict[str, object]]:
    """Run missing config/flight jobs, then collect their metrics rows."""

    rows: list[dict[str, object]] = []
    for config in configs:
        run_dir = output_dir / str(getattr(config, "name"))
        for flight in flights:
            metrics_path = metrics_json_path(run_dir, flight)
            if not (skip_existing and metrics_path.exists()):
                run_one(config, run_dir, flight)
            rows.append(make_row(config, metrics_path, load_metrics(metrics_path)))
    return rows


def run_named_config_experiments(
    args: argparse.Namespace,
    configs: Iterable[ConfigT],
    run_one: Callable[[argparse.Namespace, Path, str, ConfigT], None],
    make_row: Callable[[ConfigT, Path, dict[str, Any]], dict[str, object]],
) -> list[dict[str, object]]:
    """Collect rows for scripts whose configs map directly to output subdirectories."""

    return collect_experiment_rows(
        configs=configs,
        output_dir=args.output_dir,
        flights=args.flights,
        skip_existing=args.skip_existing,
        run_one=lambda config, run_dir, flight: run_one(args, run_dir, flight, config),
        make_row=make_row,
    )


def metrics_json_path(run_dir: Path, flight: str) -> Path:
    """Return the metrics artifact path for one run directory and flight."""

    return run_dir / flight / "metrics.json"


def load_metrics(path: Path) -> dict[str, Any]:
    """Load a metrics.json artifact."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_summary_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write rows to a CSV using the first row's key order."""

    if not rows:
        raise RuntimeError("No metrics rows were produced")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_baseline(
    *,
    dataset_root: Path,
    flight: str,
    output_dir: Path,
    association: str | None = None,
    extra_options: Iterable[object] = (),
) -> None:
    """Invoke ``raft_uav.cli run-baseline`` with repository-local src on PYTHONPATH."""

    command: list[str] = [
        sys.executable,
        "-m",
        "raft_uav.cli",
        "run-baseline",
        str(dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
    ]
    if association is not None:
        command.extend(["--radar-association", association])
    command.extend(str(option) for option in extra_options)
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=subprocess_env())


def subprocess_env() -> dict[str, str]:
    """Return an environment that imports the working tree's ``src`` package."""

    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else os.pathsep.join([src_path, current])
    return env


def robust_update_options(args: argparse.Namespace) -> list[object]:
    """Return CLI options for NIS covariance inflation."""

    return [
        "--robust-update",
        "nis-inflate",
        "--rf-gate-prob",
        args.rf_gate_prob,
        "--radar-gate-prob",
        args.radar_gate_prob,
        "--rf-inflation-alpha",
        args.rf_inflation_alpha,
        "--radar-inflation-alpha",
        args.radar_inflation_alpha,
    ]


def smoother_options(method: str, lag_s: float) -> list[object]:
    """Return CLI options for a smoother mode."""

    options: list[object] = ["--smoother", method]
    if method == "fixed-lag":
        options.extend(["--smoother-lag-s", lag_s])
    return options


def tracking_summary_row(
    method: str,
    metrics_path: Path,
    metrics: dict[str, Any],
    *,
    extra_fields: dict[str, object] | None = None,
    include_selected_track_ids: bool = False,
    include_reweighted: bool = False,
    include_inflation: bool = False,
) -> dict[str, object]:
    """Build a standard tracking-performance row from one metrics artifact."""

    robust_update = metrics.get("robust_update") or {}
    smoother = metrics.get("smoother") or {}
    row: dict[str, object] = {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": method,
        "radar_association": metrics.get("radar_association", metrics.get("radar_selection", "")),
        "robust_update": empty_if_none(robust_update.get("method")),
        "smoother": empty_if_none(smoother.get("method")),
        "smoother_lag_s": empty_if_none(smoother.get("lag_s")),
    }
    if include_inflation:
        row.update(
            {
                "rf_inflation_alpha": empty_if_none(robust_update.get("rf_inflation_alpha")),
                "radar_inflation_alpha": empty_if_none(
                    robust_update.get("radar_inflation_alpha")
                ),
            }
        )
    if extra_fields:
        row.update(extra_fields)
    row.update(
        {
            "posterior_records": int(metrics.get("posterior_records", 0)),
            "selected_radar_rows": int(metrics.get("selected_radar_rows", 0)),
        }
    )
    if include_selected_track_ids:
        row["selected_radar_track_ids"] = len(metrics.get("selected_radar_track_ids") or [])
    if include_reweighted:
        reweighted_by_source = metrics.get("reweighted_by_source") or {}
        row.update(
            {
                "reweighted_measurements": int(metrics.get("reweighted_measurements", 0)),
                "reweighted_rf": int(reweighted_by_source.get("rf", 0)),
                "reweighted_radar": int(reweighted_by_source.get("radar", 0)),
            }
        )
    row.update(error_metric_columns(metrics))
    row["metrics_path"] = str(metrics_path)
    return row


def error_metric_columns(metrics: dict[str, Any]) -> dict[str, object]:
    """Return rounded 2D/3D error metrics in CSV-column form."""

    columns: dict[str, object] = {}
    for suffix in ("2d", "3d"):
        errors = metrics.get(f"position_error_{suffix}") or {}
        for statistic in ("rmse", "mae", "p50", "p95"):
            columns[f"{statistic}_{suffix}_m"] = rounded(errors.get(f"{statistic}_m"))
    return columns


def rounded(value: object) -> object:
    """Round a numeric value to millimetre precision for summaries."""

    if value is None:
        return ""
    return round(float(value), 3)


def empty_if_none(value: object) -> object:
    """Represent ``None`` as an empty CSV field."""

    return "" if value is None else value


def slug(value: float, *, precision: int | None = None) -> str:
    """Return a filename-safe floating-point slug."""

    text = str(float(value)) if precision is None else f"{float(value):.{precision}f}"
    return text.replace("-", "m").replace(".", "p")
