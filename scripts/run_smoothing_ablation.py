"""Run fixed-lag smoothing ablations on AERPAW optimization flights."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from ablation_common import (
    add_experiment_io_arguments,
    add_fixed_lag_argument,
    add_soft_update_arguments,
    robust_update_options,
    run_baseline,
    run_named_config_experiments,
    smoother_options,
    tracking_summary_row,
    write_summary_csv,
)


@dataclass(frozen=True)
class _Config:
    name: str
    association: str
    robust: bool
    smoother: str


def main() -> int:
    parser = argparse.ArgumentParser()
    add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/smoothing_ablation"),
        default_summary_output=Path("outputs/smoothing_ablation_opt1_opt3.csv"),
    )
    add_fixed_lag_argument(parser)
    add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    configs = [
        _Config("cv_catprob", association="catprob", robust=False, smoother="none"),
        _Config("cv_prediction_nis", association="prediction-nis", robust=False, smoother="none"),
        _Config("soft_prediction_nis", association="prediction-nis", robust=True, smoother="none"),
        _Config(
            "soft_prediction_nis_fixed_lag",
            association="prediction-nis",
            robust=True,
            smoother="fixed-lag",
        ),
        _Config(
            "soft_prediction_nis_rts", association="prediction-nis", robust=True, smoother="rts"
        ),
    ]
    rows = run_named_config_experiments(args, configs, _run_one, _row)

    write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _row(config: _Config, metrics_path: Path, metrics: dict[str, object]) -> dict[str, object]:
    return tracking_summary_row(config.name, metrics_path, metrics)


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    options: list[object] = smoother_options(config.smoother, args.fixed_lag_s)
    if config.robust:
        options.extend(robust_update_options(args))
    run_baseline(
        dataset_root=args.dataset_root,
        flight=flight,
        output_dir=output_dir,
        association=config.association,
        extra_options=options,
    )


if __name__ == "__main__":
    raise SystemExit(main())
