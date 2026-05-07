"""Run PDA-mixture radar association ablations on AERPAW optimization flights."""

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
    slug,
    smoother_options,
    tracking_summary_row,
    write_summary_csv,
)


@dataclass(frozen=True)
class _Config:
    name: str
    association: str
    candidate_threshold: float
    nis_temperature: float | None = None
    catprob_exponent: float | None = None


def main() -> int:
    parser = argparse.ArgumentParser()
    add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/pda_association_ablation"),
        default_summary_output=Path("outputs/pda_association_ablation_opt1_opt3.csv"),
    )
    parser.add_argument("--candidate-thresholds", nargs="*", type=float, default=[0.4])
    parser.add_argument("--nis-temperatures", nargs="*", type=float, default=[1.0, 2.0])
    parser.add_argument("--catprob-exponents", nargs="*", type=float, default=[0.0, 0.5, 1.0])
    add_fixed_lag_argument(parser)
    add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = run_named_config_experiments(args, _configs(args), _run_one, _row)
    write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


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


def _row(config: _Config, metrics_path: Path, metrics: dict[str, object]) -> dict[str, object]:
    pda = metrics.get("pda_association") or {}
    return tracking_summary_row(
        config.name,
        metrics_path,
        metrics,
        extra_fields={
            "radar_catprob_threshold": metrics.get("radar_catprob_threshold", ""),
            "pda_nis_temperature": _metric_value(pda, "nis_temperature"),
            "pda_catprob_exponent": _metric_value(pda, "catprob_exponent"),
        },
        include_selected_track_ids=True,
    )


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    options: list[object] = ["--radar-catprob-threshold", config.candidate_threshold]
    options.extend(robust_update_options(args))
    options.extend(smoother_options("fixed-lag", args.fixed_lag_s))
    if config.association == "pda-mixture":
        options.extend(
            [
                "--pda-nis-temperature",
                config.nis_temperature,
                "--pda-catprob-exponent",
                config.catprob_exponent,
            ]
        )
    run_baseline(
        dataset_root=args.dataset_root,
        flight=flight,
        output_dir=output_dir,
        association=config.association,
        extra_options=options,
    )


def _pda_name(threshold: float, temperature: float, exponent: float) -> str:
    return (
        f"pda_mixture_t{slug(threshold, precision=2)}"
        f"_temp{slug(temperature, precision=2)}"
        f"_beta{slug(exponent, precision=2)}"
    )


def _metric_value(mapping: object, key: str) -> object:
    if not isinstance(mapping, dict):
        return ""
    from ablation_common import empty_if_none

    return empty_if_none(mapping.get(key))


if __name__ == "__main__":
    raise SystemExit(main())
