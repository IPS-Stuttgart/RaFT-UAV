"""Run radar candidate class-probability threshold ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/catprob_threshold_ablation"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/catprob_threshold_ablation.csv"),
    )
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.4, 0.5])
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for threshold in args.thresholds:
        run_name = _threshold_name(threshold)
        run_dir = args.output_dir / run_name
        for flight in args.flights:
            metrics_path = run_dir / flight / "metrics.json"
            if not (args.skip_existing and metrics_path.exists()):
                _run_one(
                    args=args,
                    output_dir=run_dir,
                    flight=flight,
                    threshold=threshold,
                )
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(_row(run_name, metrics_path, metrics))

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _run_one(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    threshold: float,
) -> None:
    command = [
        sys.executable,
        "-m",
        "raft_uav.cli",
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        "prediction-nis",
        "--radar-catprob-threshold",
        str(threshold),
        "--robust-update",
        "nis-inflate",
        "--rf-gate-prob",
        str(args.rf_gate_prob),
        "--radar-gate-prob",
        str(args.radar_gate_prob),
        "--rf-inflation-alpha",
        str(args.rf_inflation_alpha),
        "--radar-inflation-alpha",
        str(args.radar_inflation_alpha),
        "--smoother",
        "fixed-lag",
        "--smoother-lag-s",
        str(args.fixed_lag_s),
    ]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=_subprocess_env())


def _row(method: str, metrics_path: Path, metrics: dict[str, Any]) -> dict[str, object]:
    error_2d = metrics.get("position_error_2d") or {}
    error_3d = metrics.get("position_error_3d") or {}
    robust_update = metrics.get("robust_update") or {}
    smoother = metrics.get("smoother") or {}
    return {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": method,
        "radar_association": metrics.get("radar_association", metrics.get("radar_selection", "")),
        "radar_catprob_threshold": metrics.get("radar_catprob_threshold", ""),
        "robust_update": _empty_if_none(robust_update.get("method")),
        "smoother": _empty_if_none(smoother.get("method")),
        "smoother_lag_s": _empty_if_none(smoother.get("lag_s")),
        "posterior_records": int(metrics.get("posterior_records", 0)),
        "selected_radar_rows": int(metrics.get("selected_radar_rows", 0)),
        "selected_radar_track_ids": len(metrics.get("selected_radar_track_ids") or []),
        "rmse_2d_m": _rounded(error_2d.get("rmse_m")),
        "mae_2d_m": _rounded(error_2d.get("mae_m")),
        "p50_2d_m": _rounded(error_2d.get("p50_m")),
        "p95_2d_m": _rounded(error_2d.get("p95_m")),
        "rmse_3d_m": _rounded(error_3d.get("rmse_m")),
        "mae_3d_m": _rounded(error_3d.get("mae_m")),
        "p50_3d_m": _rounded(error_3d.get("p50_m")),
        "p95_3d_m": _rounded(error_3d.get("p95_m")),
        "metrics_path": str(metrics_path),
    }


def _threshold_name(threshold: float) -> str:
    return f"prediction_nis_t{_slug(threshold)}"


def _slug(value: float) -> str:
    return f"{float(value):.2f}".replace("-", "m").replace(".", "p")


def _rounded(value: object) -> object:
    if value is None:
        return ""
    return round(float(value), 3)


def _empty_if_none(value: object) -> object:
    return "" if value is None else value


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(REPO_ROOT / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = src_path if not current else os.pathsep.join([src_path, current])
    return env


if __name__ == "__main__":
    raise SystemExit(main())
