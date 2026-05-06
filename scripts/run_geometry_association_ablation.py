"""Run a targeted geometry-score association ablation."""

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
        default=Path("outputs/geometry_association_ablation"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/geometry_association_ablation.csv"),
    )
    parser.add_argument("--flights", nargs="*", default=["Opt1"])
    parser.add_argument("--velocity-weights", nargs="*", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--switch-penalties", nargs="*", type=float, default=[0.0, 4.0, 8.0])
    parser.add_argument("--catprob-weights", nargs="*", type=float, default=[0.0, 2.0])
    parser.add_argument("--geometry-velocity-std", type=float, default=12.0)
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    configs = [
        _Config(
            "soft_prediction_nis_fixed_lag",
            association="prediction-nis",
            velocity_weight=None,
            switch_penalty=None,
            catprob_weight=None,
        )
    ]
    for velocity_weight in args.velocity_weights:
        for switch_penalty in args.switch_penalties:
            for catprob_weight in args.catprob_weights:
                configs.append(
                    _Config(
                        _geometry_name(velocity_weight, switch_penalty, catprob_weight),
                        association="geometry-score",
                        velocity_weight=velocity_weight,
                        switch_penalty=switch_penalty,
                        catprob_weight=catprob_weight,
                    )
                )

    rows: list[dict[str, object]] = []
    for config in configs:
        run_dir = args.output_dir / config.name
        for flight in args.flights:
            metrics_path = run_dir / flight / "metrics.json"
            if not (args.skip_existing and metrics_path.exists()):
                _run_one(args=args, output_dir=run_dir, flight=flight, config=config)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append(_row(config.name, metrics_path, metrics))

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


class _Config:
    def __init__(
        self,
        name: str,
        *,
        association: str,
        velocity_weight: float | None,
        switch_penalty: float | None,
        catprob_weight: float | None,
    ) -> None:
        self.name = name
        self.association = association
        self.velocity_weight = velocity_weight
        self.switch_penalty = switch_penalty
        self.catprob_weight = catprob_weight


def _run_one(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
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
        config.association,
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
    if config.association == "geometry-score":
        command.extend(
            [
                "--geometry-velocity-std",
                str(args.geometry_velocity_std),
                "--geometry-velocity-weight",
                str(config.velocity_weight),
                "--geometry-switch-penalty",
                str(config.switch_penalty),
                "--geometry-catprob-weight",
                str(config.catprob_weight),
            ]
        )
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=_subprocess_env())


def _row(method: str, metrics_path: Path, metrics: dict[str, Any]) -> dict[str, object]:
    error_2d = metrics.get("position_error_2d") or {}
    error_3d = metrics.get("position_error_3d") or {}
    robust_update = metrics.get("robust_update") or {}
    smoother = metrics.get("smoother") or {}
    geometry = metrics.get("geometry_association") or {}
    return {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": method,
        "radar_association": metrics.get("radar_association", metrics.get("radar_selection", "")),
        "robust_update": _empty_if_none(robust_update.get("method")),
        "smoother": _empty_if_none(smoother.get("method")),
        "smoother_lag_s": _empty_if_none(smoother.get("lag_s")),
        "geometry_velocity_std_mps": _empty_if_none(geometry.get("velocity_std_mps")),
        "geometry_velocity_weight": _empty_if_none(geometry.get("velocity_weight")),
        "geometry_switch_penalty": _empty_if_none(geometry.get("switch_penalty")),
        "geometry_catprob_weight": _empty_if_none(geometry.get("catprob_weight")),
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


def _geometry_name(
    velocity_weight: float,
    switch_penalty: float,
    catprob_weight: float,
) -> str:
    return (
        "geometry_score"
        f"_v{_slug(velocity_weight)}"
        f"_s{_slug(switch_penalty)}"
        f"_c{_slug(catprob_weight)}"
    )


def _slug(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


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
