"""Run a source-specific NIS covariance-inflation alpha grid."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_common import (
    add_experiment_io_arguments,
    error_metric_columns,
    load_metrics,
    metrics_json_path,
    run_baseline,
    slug,
    write_summary_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/source_specific_grid"),
        default_summary_output=Path("outputs/source_specific_inflation_grid_opt1_opt3.csv"),
    )
    parser.add_argument("--rf-alphas", nargs="*", type=float, default=[0.5, 1.0, 1.5, 2.0])
    parser.add_argument("--radar-alphas", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    parser.add_argument("--rf-gate-prob", type=float, default=0.99)
    parser.add_argument("--radar-gate-prob", type=float, default=0.99)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for rf_alpha in args.rf_alphas:
        for radar_alpha in args.radar_alphas:
            rows.extend(_rows_for_alpha_pair(args, rf_alpha, radar_alpha))

    write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _rows_for_alpha_pair(
    args: argparse.Namespace,
    rf_alpha: float,
    radar_alpha: float,
) -> list[dict[str, object]]:
    run_dir = args.output_dir / _combo_name(rf_alpha, radar_alpha)
    rows: list[dict[str, object]] = []
    for flight in args.flights:
        metrics_path = metrics_json_path(run_dir, flight)
        if not (args.skip_existing and metrics_path.exists()):
            _run_one(args, run_dir, flight, rf_alpha, radar_alpha)
        rows.append(_row(metrics_path, load_metrics(metrics_path), rf_alpha, radar_alpha))
    return rows


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    rf_alpha: float,
    radar_alpha: float,
) -> None:
    run_baseline(
        dataset_root=args.dataset_root,
        flight=flight,
        output_dir=output_dir,
        extra_options=[
            "--robust-update",
            "nis-inflate",
            "--rf-gate-prob",
            args.rf_gate_prob,
            "--radar-gate-prob",
            args.radar_gate_prob,
            "--rf-inflation-alpha",
            rf_alpha,
            "--radar-inflation-alpha",
            radar_alpha,
        ],
    )


def _row(
    metrics_path: Path,
    metrics: dict[str, object],
    rf_alpha: float,
    radar_alpha: float,
) -> dict[str, object]:
    reweighted_by_source = metrics.get("reweighted_by_source") or {}
    row = {
        "flight": metrics.get("flight", metrics_path.parent.name),
        "method": "cv_nis_inflated",
        "rf_inflation_alpha": rf_alpha,
        "radar_inflation_alpha": radar_alpha,
        "posterior_records": int(metrics.get("posterior_records", 0)),
        "reweighted_measurements": int(metrics.get("reweighted_measurements", 0)),
        "reweighted_rf": _source_count(reweighted_by_source, "rf"),
        "reweighted_radar": _source_count(reweighted_by_source, "radar"),
    }
    row.update(error_metric_columns(metrics))
    row["metrics_path"] = str(metrics_path)
    return row


def _source_count(counts: object, source: str) -> int:
    return int(counts.get(source, 0)) if isinstance(counts, dict) else 0


def _combo_name(rf_alpha: float, radar_alpha: float) -> str:
    return f"rf{slug(rf_alpha)}_radar{slug(radar_alpha)}"


if __name__ == "__main__":
    raise SystemExit(main())
