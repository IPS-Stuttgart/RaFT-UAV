"""Run radar association ablations on AERPAW optimization flights."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/radar_association_ablation"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/radar_association_ablation_opt1_opt3.csv"),
    )
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    parser.add_argument(
        "--associations",
        nargs="*",
        default=["catprob", "oracle-nearest-truth", "prediction-nis", "track-continuity"],
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        choices=["cv", "soft"],
        default=["cv", "soft"],
        help="cv is ungated; soft is NIS covariance inflation",
    )
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--track-switch-nis-ratio", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for variant in args.variants:
        for association in args.associations:
            run_name = f"{variant}_{association}"
            run_dir = args.output_dir / run_name
            for flight in args.flights:
                metrics_path = run_dir / flight / "metrics.json"
                if not (args.skip_existing and metrics_path.exists()):
                    _run_one(
                        dataset_root=args.dataset_root,
                        output_dir=run_dir,
                        flight=flight,
                        association=association,
                        variant=variant,
                        rf_gate_prob=args.rf_gate_prob,
                        radar_gate_prob=args.radar_gate_prob,
                        rf_inflation_alpha=args.rf_inflation_alpha,
                        radar_inflation_alpha=args.radar_inflation_alpha,
                        track_switch_nis_ratio=args.track_switch_nis_ratio,
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
    dataset_root: Path,
    output_dir: Path,
    flight: str,
    association: str,
    variant: str,
    rf_gate_prob: float,
    radar_gate_prob: float,
    rf_inflation_alpha: float,
    radar_inflation_alpha: float,
    track_switch_nis_ratio: float,
) -> None:
    command = [
        sys.executable,
        "-m",
        "raft_uav.cli",
        "run-baseline",
        str(dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        association,
        "--track-switch-nis-ratio",
        str(track_switch_nis_ratio),
    ]
    if variant == "soft":
        command.extend(
            [
                "--robust-update",
                "nis-inflate",
                "--rf-gate-prob",
                str(rf_gate_prob),
                "--radar-gate-prob",
                str(radar_gate_prob),
                "--rf-inflation-alpha",
                str(rf_inflation_alpha),
                "--radar-inflation-alpha",
                str(radar_inflation_alpha),
            ]
        )
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def _row(method: str, metrics_path: Path, metrics: dict[str, Any]) -> dict[str, object]:
    error_2d = metrics.get("position_error_2d") or {}
    error_3d = metrics.get("position_error_3d") or {}
    robust_update = metrics.get("robust_update") or {}
    reweighted_by_source = metrics.get("reweighted_by_source") or {}
    return {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": method,
        "radar_association": metrics.get("radar_association", metrics.get("radar_selection", "")),
        "robust_update": _empty_if_none(robust_update.get("method")),
        "rf_inflation_alpha": _empty_if_none(robust_update.get("rf_inflation_alpha")),
        "radar_inflation_alpha": _empty_if_none(robust_update.get("radar_inflation_alpha")),
        "posterior_records": int(metrics.get("posterior_records", 0)),
        "selected_radar_rows": int(metrics.get("selected_radar_rows", 0)),
        "selected_radar_track_ids": len(metrics.get("selected_radar_track_ids") or []),
        "reweighted_measurements": int(metrics.get("reweighted_measurements", 0)),
        "reweighted_rf": int(reweighted_by_source.get("rf", 0)),
        "reweighted_radar": int(reweighted_by_source.get("radar", 0)),
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


def _rounded(value: object) -> object:
    if value is None:
        return ""
    return round(float(value), 3)


def _empty_if_none(value: object) -> object:
    return "" if value is None else value


if __name__ == "__main__":
    raise SystemExit(main())
