"""Run a targeted geometry-score association ablation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import ablation_common as common


@dataclass(frozen=True)
class _Config:
    name: str
    association: str
    velocity_weight: float | None
    switch_penalty: float | None
    catprob_weight: float | None


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/geometry_association_ablation"),
        default_summary_output=Path("outputs/geometry_association_ablation.csv"),
        default_flights=["Opt1"],
    )
    parser.add_argument("--velocity-weights", nargs="*", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--switch-penalties", nargs="*", type=float, default=[0.0, 4.0, 8.0])
    parser.add_argument("--catprob-weights", nargs="*", type=float, default=[0.0, 2.0])
    parser.add_argument("--geometry-velocity-std", type=float, default=12.0)
    common.add_fixed_lag_argument(parser)
    common.add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    rows = common.run_named_config_experiments(args, _configs(args), _run_one, _row)
    common.write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _configs(args: argparse.Namespace) -> list[_Config]:
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
    return configs


def _row(config: _Config, metrics_path: Path, metrics: dict[str, object]) -> dict[str, object]:
    geometry = metrics.get("geometry_association") or {}
    return common.tracking_summary_row(
        config.name,
        metrics_path,
        metrics,
        extra_fields={
            "geometry_velocity_std_mps": _metric_value(geometry, "velocity_std_mps"),
            "geometry_velocity_weight": _metric_value(geometry, "velocity_weight"),
            "geometry_switch_penalty": _metric_value(geometry, "switch_penalty"),
            "geometry_catprob_weight": _metric_value(geometry, "catprob_weight"),
        },
        include_selected_track_ids=True,
    )


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    options: list[object] = []
    options.extend(common.robust_update_options(args))
    options.extend(common.smoother_options("fixed-lag", args.fixed_lag_s))
    if config.association == "geometry-score":
        options.extend(
            [
                "--geometry-velocity-std",
                args.geometry_velocity_std,
                "--geometry-velocity-weight",
                config.velocity_weight,
                "--geometry-switch-penalty",
                config.switch_penalty,
                "--geometry-catprob-weight",
                config.catprob_weight,
            ]
        )
    common.run_baseline(
        dataset_root=args.dataset_root,
        flight=flight,
        output_dir=output_dir,
        association=config.association,
        extra_options=options,
    )


def _geometry_name(
    velocity_weight: float,
    switch_penalty: float,
    catprob_weight: float,
) -> str:
    return (
        "geometry_score"
        f"_v{common.slug(velocity_weight)}"
        f"_s{common.slug(switch_penalty)}"
        f"_c{common.slug(catprob_weight)}"
    )


def _metric_value(mapping: object, key: str) -> object:
    return common.empty_if_none(mapping.get(key)) if isinstance(mapping, dict) else ""


if __name__ == "__main__":
    raise SystemExit(main())
