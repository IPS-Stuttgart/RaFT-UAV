"""Run radar association ablations on AERPAW optimization flights."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_common import (
    add_experiment_io_arguments,
    add_soft_update_arguments,
    load_metrics,
    metrics_json_path,
    robust_update_options,
    run_baseline,
    tracking_summary_row,
    write_summary_csv,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/radar_association_ablation"),
        default_summary_output=Path("outputs/radar_association_ablation_opt1_opt3.csv"),
    )
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
    add_soft_update_arguments(parser)
    parser.add_argument("--track-switch-nis-ratio", type=float, default=0.5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for variant in args.variants:
        for association in args.associations:
            rows.extend(_rows_for_run(args, variant, association))

    write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _rows_for_run(
    args: argparse.Namespace,
    variant: str,
    association: str,
) -> list[dict[str, object]]:
    run_name = f"{variant}_{association}"
    run_dir = args.output_dir / run_name
    rows: list[dict[str, object]] = []
    for flight in args.flights:
        metrics_path = metrics_json_path(run_dir, flight)
        if not (args.skip_existing and metrics_path.exists()):
            _run_one(args, run_dir, flight, association, variant)
        metrics = load_metrics(metrics_path)
        rows.append(
            tracking_summary_row(
                run_name,
                metrics_path,
                metrics,
                include_selected_track_ids=True,
                include_reweighted=True,
                include_inflation=True,
            )
        )
    return rows


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    association: str,
    variant: str,
) -> None:
    options: list[object] = ["--track-switch-nis-ratio", args.track_switch_nis_ratio]
    if variant == "soft":
        options.extend(robust_update_options(args))
    run_baseline(
        dataset_root=args.dataset_root,
        flight=flight,
        output_dir=output_dir,
        association=association,
        extra_options=options,
    )


if __name__ == "__main__":
    raise SystemExit(main())
