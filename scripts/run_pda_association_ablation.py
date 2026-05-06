"""Run PDA-mixture radar association ablations on AERPAW optimization flights."""

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
        default=Path("outputs/pda_association_ablation"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/pda_association_ablation_opt1_opt3.csv"),
    )
    parser.add_argument("--flights", nargs="*", default=["Opt1", "Opt2", "Opt3"])
    parser.add_argument("--candidate-thresholds", nargs="*", type=float, default=[0.4])
    parser.add_argument("--nis-temperatures", nargs="*", type=float, default=[1.0, 2.0])
    parser.add_argument("--catprob-exponents", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for config in _configs(args):
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
        candidate_threshold: float,
        nis_temperature: float | None = None,
        catprob_exponent: float | None = None,
    ) -> None:
        self.name = name
        self.association = association
        self.candidate_threshold = candidate_threshold
        self.nis_temperature = nis_temperature
        self.catprob_exponent = catprob_exponent


def _configs(args: argparse.Namespace) -> list[_Config]:
    configs = [
        _Config(
            "prediction_nis_t0p40",
            association="prediction-nis",
            candidate_threshold=0.4,
        )
    ]
    for threshold in args.candidate_thresholds:
        for temperature in args.nis_temperatures:
            for exponent in args.catprob_exponents:
                configs.append(
                    _Config(
                        _pda_name(threshold, temperature, exponent),
                        association="pda-mixture",
                        candidate_threshold=threshold,
                        nis_temperature=temperature,
                        catprob_exponent=exponent,
                    )
                )
    return configs


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
        "--radar-catprob-threshold",
        str(config.candidate_threshold),
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
    if config.association == "pda-mixture":
        command.extend(
            [
                "--pda-nis-temperature",
                str(config.nis_temperature),
                "--pda-catprob-exponent",
                str(config.catprob_exponent),
            ]
        )
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=_subprocess_env())


def _row(method: str, metrics_path: Path, metrics: dict[str, Any]) -> dict[str, object]:
    error_2d = metrics.get("position_error_2d") or {}
    error_3d = metrics.get("position_error_3d") or {}
    robust_update = metrics.get("robust_update") or {}
    smoother = metrics.get("smoother") or {}
    pda = metrics.get("pda_association") or {}
    return {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": method,
        "radar_association": metrics.get("radar_association", metrics.get("radar_selection", "")),
        "radar_catprob_threshold": metrics.get("radar_catprob_threshold", ""),
        "pda_nis_temperature": _empty_if_none(pda.get("nis_temperature")),
        "pda_catprob_exponent": _empty_if_none(pda.get("catprob_exponent")),
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


def _pda_name(threshold: float, temperature: float, exponent: float) -> str:
    return (
        f"pda_mixture_t{_slug(threshold)}"
        f"_temp{_slug(temperature)}"
        f"_beta{_slug(exponent)}"
    )


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
