"""Grid-search runner for tracklet-Viterbi LOFO experiments.

The runner launches ``run_tracklet_viterbi_lofo.py`` for each association-cost
configuration, optionally augments the result with full replay diagnostics, and
writes a rankable summary table.  It is intentionally resumable: each grid point
gets a stable output directory and existing aggregate summaries are reused when
``--skip-existing`` is set.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ablation_common as common  # noqa: E402


@dataclass(frozen=True)
class GridConfig:
    """One tracklet-Viterbi cost configuration."""

    anchor_nis_weight: float
    track_switch_cost: float
    missed_detection_cost: float
    velocity_nis_weight: float
    range_gate_m: float | None

    @property
    def config_id(self) -> str:
        """Stable filesystem-safe identifier."""

        parts = {
            "a": self.anchor_nis_weight,
            "sw": self.track_switch_cost,
            "miss": self.missed_detection_cost,
            "vel": self.velocity_nis_weight,
            "rg": "none" if self.range_gate_m is None else self.range_gate_m,
        }
        readable = "_".join(f"{key}{_slug(value)}" for key, value in parts.items())
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        digest = hashlib.sha1(payload).hexdigest()[:8]
        return f"{readable}_{digest}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tracklet_viterbi_grid"))
    parser.add_argument("--flights", nargs="*", default=None)
    parser.add_argument("--candidate-threshold", type=float, default=0.4)
    parser.add_argument("--fixed-lag-s", type=float, default=20.0)
    parser.add_argument("--max-eval-time-delta-s", type=float, default=2.0)
    parser.add_argument("--acceleration-std", type=float, default=4.0)
    parser.add_argument("--max-candidates-per-frame", type=int, default=8)
    parser.add_argument("--catprob-weight", type=float, default=2.5)
    parser.add_argument("--transition-nis-weight", type=float, default=1.0)
    parser.add_argument("--max-speed-mps", type=float, default=55.0)
    parser.add_argument("--disable-rf-anchor", action="store_true")
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--rf-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--radar-safety-gate-prob", type=float, default=0.9999999)
    parser.add_argument("--rf-max-residual-m", type=float, default=750.0)
    parser.add_argument("--radar-max-residual-m", type=float, default=0.0)
    parser.add_argument("--robust-update", choices=["none", "nis-inflate"], default="nis-inflate")
    parser.add_argument("--rf-inflation-alpha", type=float, default=0.5)
    parser.add_argument("--radar-inflation-alpha", type=float, default=0.5)

    parser.add_argument("--anchor-nis-weights", nargs="+", type=float, default=[0.15, 0.35, 0.70])
    parser.add_argument("--track-switch-costs", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    parser.add_argument("--missed-detection-costs", nargs="+", type=float, default=[4.0, 7.0, 12.0])
    parser.add_argument("--velocity-nis-weights", nargs="+", type=float, default=[0.0, 0.15, 0.35])
    parser.add_argument("--range-gates-m", nargs="+", default=["700", "850", "none"])

    parser.add_argument("--score-p95-weight", type=float, default=0.5)
    parser.add_argument("--score-viterbi-p95-weight", type=float, default=0.5)
    parser.add_argument("--score-coverage-target", type=float, default=0.98)
    parser.add_argument("--score-coverage-penalty", type=float, default=100.0)
    parser.add_argument("--sort-by", default="score")
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-augment-replay", action="store_true")
    args = parser.parse_args(argv)

    configs = list(_grid_configs(args))
    if args.limit_configs is not None:
        configs = configs[: args.limit_configs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "grid_config.json", _grid_metadata(args, configs))

    rows: list[dict[str, object]] = []
    for index, config in enumerate(configs, start=1):
        config_dir = args.output_dir / config.config_id
        aggregate_path = _aggregate_path(config_dir, augment_replay=not args.no_augment_replay)
        print(f"[{index}/{len(configs)}] {config.config_id}", flush=True)
        if args.dry_run:
            print("  dry-run: " + " ".join(_lofo_command(args, config, config_dir)), flush=True)
            rows.append(_dry_run_row(config, index, len(configs), config_dir))
            continue
        if not (args.skip_existing and aggregate_path.exists()):
            _run(_lofo_command(args, config, config_dir))
            if not args.no_augment_replay:
                _run(_augment_command(args, config_dir))
        rows.append(_summary_row(config, index, len(configs), config_dir, aggregate_path, args))

    ranked = _rank_rows(rows, sort_by=args.sort_by)
    _write_csv(args.output_dir / "grid_results.csv", ranked)
    _write_csv(args.output_dir / "grid_results_ranked.csv", ranked)
    _write_json(args.output_dir / "grid_results_ranked.json", ranked)
    if ranked:
        print(f"best {args.sort_by}: {ranked[0]['config_id']} = {ranked[0].get(args.sort_by)}")
    print(f"wrote {len(ranked)} rows to {args.output_dir / 'grid_results_ranked.csv'}")
    return 0


def _grid_configs(args: argparse.Namespace) -> Iterable[GridConfig]:
    ranges = [_parse_optional_float(value) for value in args.range_gates_m]
    for anchor, switch, missed, velocity, range_gate in itertools.product(
        args.anchor_nis_weights,
        args.track_switch_costs,
        args.missed_detection_costs,
        args.velocity_nis_weights,
        ranges,
    ):
        yield GridConfig(
            anchor_nis_weight=float(anchor),
            track_switch_cost=float(switch),
            missed_detection_cost=float(missed),
            velocity_nis_weight=float(velocity),
            range_gate_m=range_gate,
        )


def _lofo_command(args: argparse.Namespace, config: GridConfig, config_dir: Path) -> list[str]:
    command: list[object] = [
        sys.executable,
        "scripts/run_tracklet_viterbi_lofo.py",
        args.dataset_root,
        "--output-dir",
        config_dir,
        "--candidate-threshold",
        args.candidate_threshold,
        "--fixed-lag-s",
        args.fixed_lag_s,
        "--max-eval-time-delta-s",
        args.max_eval_time_delta_s,
        "--acceleration-std",
        args.acceleration_std,
        "--max-candidates-per-frame",
        args.max_candidates_per_frame,
        "--missed-detection-cost",
        config.missed_detection_cost,
        "--track-switch-cost",
        config.track_switch_cost,
        "--catprob-weight",
        args.catprob_weight,
        "--anchor-nis-weight",
        config.anchor_nis_weight,
        "--transition-nis-weight",
        args.transition_nis_weight,
        "--velocity-nis-weight",
        config.velocity_nis_weight,
        "--max-speed-mps",
        args.max_speed_mps,
        "--range-gate-m",
        0.0 if config.range_gate_m is None else config.range_gate_m,
        "--rf-gate-prob",
        args.rf_gate_prob,
        "--radar-gate-prob",
        args.radar_gate_prob,
        "--rf-safety-gate-prob",
        args.rf_safety_gate_prob,
        "--radar-safety-gate-prob",
        args.radar_safety_gate_prob,
        "--rf-max-residual-m",
        args.rf_max_residual_m,
        "--radar-max-residual-m",
        args.radar_max_residual_m,
        "--robust-update",
        args.robust_update,
        "--rf-inflation-alpha",
        args.rf_inflation_alpha,
        "--radar-inflation-alpha",
        args.radar_inflation_alpha,
        "--skip-existing",
    ]
    if args.flights:
        command.extend(["--flights", *args.flights])
    if args.disable_rf_anchor:
        command.append("--disable-rf-anchor")
    return [str(item) for item in command]


def _augment_command(args: argparse.Namespace, config_dir: Path) -> list[str]:
    command: list[object] = [
        sys.executable,
        "scripts/augment_tracklet_viterbi_lofo_replay_summary.py",
        args.dataset_root,
        "--summary-dir",
        config_dir,
        "--max-eval-time-delta-s",
        args.max_eval_time_delta_s,
    ]
    return [str(item) for item in command]


def _summary_row(
    config: GridConfig,
    index: int,
    total: int,
    config_dir: Path,
    aggregate_path: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    if not aggregate_path.exists():
        raise FileNotFoundError(
            f"missing aggregate summary for {config.config_id}: {aggregate_path}"
        )
    aggregate = _read_single_csv_row(aggregate_path)
    row: dict[str, object] = {
        "rank": None,
        "grid_index": index,
        "grid_size": total,
        "config_id": config.config_id,
        "output_dir": str(config_dir),
        "aggregate_path": str(aggregate_path),
        **asdict(config),
    }
    row.update({f"metric_{key}": value for key, value in aggregate.items()})
    row.update(_score_columns(aggregate, args))
    return row


def _score_columns(aggregate: dict[str, object], args: argparse.Namespace) -> dict[str, float]:
    rmse = _float_metric(aggregate, "error_3d_rmse_m")
    p95 = _float_metric(aggregate, "error_3d_p95_m")
    viterbi_p95 = _float_metric(aggregate, "viterbi_selected_radar_error_3d_p95_m")
    coverage = _float_metric(aggregate, "truth_coverage_rate")
    if not np_isfinite(coverage):
        coverage = _float_metric(aggregate, "viterbi_selected_radar_truth_coverage_rate")
    coverage_shortfall = (
        max(0.0, float(args.score_coverage_target) - coverage)
        if np_isfinite(coverage)
        else 1.0
    )
    score = (
        rmse
        + float(args.score_p95_weight) * p95
        + float(args.score_viterbi_p95_weight) * viterbi_p95
        + float(args.score_coverage_penalty) * coverage_shortfall
    )
    return {
        "score": score,
        "score_rmse_3d_m": rmse,
        "score_p95_3d_m": p95,
        "score_viterbi_p95_3d_m": viterbi_p95,
        "score_coverage_rate": coverage,
        "score_coverage_shortfall": coverage_shortfall,
    }


def _rank_rows(rows: list[dict[str, object]], *, sort_by: str) -> list[dict[str, object]]:
    def key(row: dict[str, object]) -> tuple[bool, float]:
        value = _optional_float(row.get(sort_by))
        if value is None:
            return True, float("inf")
        return False, value

    ranked = sorted(rows, key=key)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _aggregate_path(config_dir: Path, *, augment_replay: bool) -> Path:
    if augment_replay:
        return config_dir / "aggregate_viterbi_replay_summary.csv"
    return config_dir / "aggregate_summary.csv"


def _read_single_csv_row(path: Path) -> dict[str, object]:
    rows = _read_csv_rows(path)
    if len(rows) != 1:
        raise ValueError(f"expected exactly one row in {path}, got {len(rows)}")
    return rows[0]


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _grid_metadata(args: argparse.Namespace, configs: Sequence[GridConfig]) -> dict[str, object]:
    return {
        "dataset_root": str(args.dataset_root),
        "output_dir": str(args.output_dir),
        "flights": args.flights,
        "candidate_threshold": args.candidate_threshold,
        "fixed_lag_s": args.fixed_lag_s,
        "score": {
            "p95_weight": args.score_p95_weight,
            "viterbi_p95_weight": args.score_viterbi_p95_weight,
            "coverage_target": args.score_coverage_target,
            "coverage_penalty": args.score_coverage_penalty,
        },
        "configs": [asdict(config) | {"config_id": config.config_id} for config in configs],
    }


def _dry_run_row(config: GridConfig, index: int, total: int, config_dir: Path) -> dict[str, object]:
    return {
        "rank": index,
        "grid_index": index,
        "grid_size": total,
        "config_id": config.config_id,
        "output_dir": str(config_dir),
        **asdict(config),
        "score": float("nan"),
    }


def _parse_optional_float(value: object) -> float | None:
    text = str(value).strip().lower()
    if text in {"none", "null", "nan", "off", "disabled"}:
        return None
    parsed = float(text)
    return None if parsed <= 0.0 else parsed


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np_isfinite(parsed) else None


def _float_metric(row: dict[str, object], key: str) -> float:
    value = _optional_float(row.get(key))
    if value is None:
        return float("inf")
    return value


def np_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _slug(value: object) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace("+", "")


def _run(command: Sequence[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=common.subprocess_env())


if __name__ == "__main__":
    raise SystemExit(main())
