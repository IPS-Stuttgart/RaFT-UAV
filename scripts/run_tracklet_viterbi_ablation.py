"""Run sequence-level tracklet-Viterbi radar association ablations."""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ablation_common as common  # noqa: E402
from raft_uav.tracklet_viterbi_cli import main as tracklet_viterbi_main  # noqa: E402


@dataclass(frozen=True)
class _Config:
    name: str
    threshold: float
    max_candidates_per_frame: int
    max_candidate_pool_per_frame: int
    max_candidates_per_track_id: int
    below_catprob_threshold_penalty: float
    track_support_weight: float
    max_track_support_reward: float


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/tracklet_viterbi_ablation"),
        default_summary_output=Path("outputs/tracklet_viterbi_ablation.csv"),
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.3, 0.4, 0.5])
    parser.add_argument("--tracklet-max-candidates-per-frame", nargs="*", type=int, default=[8])
    parser.add_argument(
        "--tracklet-max-candidate-pool-per-frame",
        nargs="*",
        type=int,
        default=[16],
    )
    parser.add_argument("--tracklet-max-candidates-per-track-id", nargs="*", type=int, default=[1])
    parser.add_argument(
        "--tracklet-below-catprob-threshold-penalties",
        nargs="*",
        type=float,
        default=[3.0],
    )
    parser.add_argument(
        "--tracklet-track-support-weights",
        nargs="*",
        type=float,
        default=[0.0, 0.45, 0.75],
    )
    parser.add_argument(
        "--tracklet-max-track-support-rewards",
        nargs="*",
        type=float,
        default=[4.0],
    )
    common.add_fixed_lag_argument(parser)
    common.add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    _validate_args(args)

    configs = _configs(args)
    rows = common.run_named_config_experiments(args, configs, _run_one, _candidate_row)
    common.write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


def _configs(args: argparse.Namespace) -> list[_Config]:
    configs: list[_Config] = []
    for values in itertools.product(
        args.thresholds,
        args.tracklet_max_candidates_per_frame,
        args.tracklet_max_candidate_pool_per_frame,
        args.tracklet_max_candidates_per_track_id,
        args.tracklet_below_catprob_threshold_penalties,
        args.tracklet_track_support_weights,
        args.tracklet_max_track_support_rewards,
    ):
        config = _Config(
            name="",
            threshold=float(values[0]),
            max_candidates_per_frame=int(values[1]),
            max_candidate_pool_per_frame=int(values[2]),
            max_candidates_per_track_id=int(values[3]),
            below_catprob_threshold_penalty=float(values[4]),
            track_support_weight=float(values[5]),
            max_track_support_reward=float(values[6]),
        )
        configs.append(_Config(name=_config_name(config), **_config_values(config)))
    return configs


def _config_values(config: _Config) -> dict[str, object]:
    return {
        "threshold": config.threshold,
        "max_candidates_per_frame": config.max_candidates_per_frame,
        "max_candidate_pool_per_frame": config.max_candidate_pool_per_frame,
        "max_candidates_per_track_id": config.max_candidates_per_track_id,
        "below_catprob_threshold_penalty": config.below_catprob_threshold_penalty,
        "track_support_weight": config.track_support_weight,
        "max_track_support_reward": config.max_track_support_reward,
    }


def _candidate_row(
    config: _Config,
    metrics_path: Path,
    metrics: dict[str, object],
) -> dict[str, object]:
    return common.tracking_summary_row(
        config.name,
        metrics_path,
        metrics,
        extra_fields={
            "radar_catprob_threshold": metrics.get("radar_catprob_threshold", ""),
            "tracklet_max_candidates_per_frame": config.max_candidates_per_frame,
            "tracklet_max_candidate_pool_per_frame": config.max_candidate_pool_per_frame,
            "tracklet_max_candidates_per_track_id": config.max_candidates_per_track_id,
            "tracklet_below_catprob_threshold_penalty": config.below_catprob_threshold_penalty,
            "tracklet_track_support_weight": config.track_support_weight,
            "tracklet_max_track_support_reward": config.max_track_support_reward,
        },
        include_selected_track_ids=True,
        include_reweighted=True,
        include_inflation=True,
    )


def _run_one(
    args: argparse.Namespace,
    output_dir: Path,
    flight: str,
    config: _Config,
) -> None:
    cli_args = [
        "run-baseline",
        str(args.dataset_root),
        "--flight",
        flight,
        "--output-dir",
        str(output_dir),
        "--radar-association",
        "tracklet-viterbi",
        "--radar-catprob-threshold",
        str(config.threshold),
        "--tracklet-max-candidates-per-frame",
        str(config.max_candidates_per_frame),
        "--tracklet-max-candidate-pool-per-frame",
        str(config.max_candidate_pool_per_frame),
        "--tracklet-max-candidates-per-track-id",
        str(config.max_candidates_per_track_id),
        "--tracklet-below-catprob-threshold-penalty",
        str(config.below_catprob_threshold_penalty),
        "--tracklet-track-support-weight",
        str(config.track_support_weight),
        "--tracklet-max-track-support-reward",
        str(config.max_track_support_reward),
        *[str(option) for option in common.robust_update_options(args)],
        *[str(option) for option in common.smoother_options("fixed-lag", args.fixed_lag_s)],
    ]
    print("raft-uav-tracklet-viterbi " + " ".join(cli_args), flush=True)
    status = tracklet_viterbi_main(cli_args)
    if status != 0:
        raise RuntimeError(f"tracklet-viterbi run failed with status {status}")


def _config_name(config: _Config) -> str:
    return "_".join(
        [
            f"tracklet_t{common.slug(config.threshold, precision=2)}",
            f"k{config.max_candidates_per_frame}",
            f"pool{config.max_candidate_pool_per_frame}",
            f"perid{config.max_candidates_per_track_id}",
            f"catpen{common.slug(config.below_catprob_threshold_penalty, precision=1)}",
            f"sw{common.slug(config.track_support_weight, precision=2)}",
            f"sr{common.slug(config.max_track_support_reward, precision=1)}",
        ]
    )


def _validate_args(args: argparse.Namespace) -> None:
    _require_nonempty("--thresholds", args.thresholds)
    _require_positive_ints(
        "--tracklet-max-candidates-per-frame",
        args.tracklet_max_candidates_per_frame,
    )
    _require_positive_ints(
        "--tracklet-max-candidate-pool-per-frame",
        args.tracklet_max_candidate_pool_per_frame,
    )
    _require_positive_ints(
        "--tracklet-max-candidates-per-track-id",
        args.tracklet_max_candidates_per_track_id,
    )
    _require_nonnegative_floats(
        "--tracklet-below-catprob-threshold-penalties",
        args.tracklet_below_catprob_threshold_penalties,
    )
    _require_nonnegative_floats(
        "--tracklet-track-support-weights",
        args.tracklet_track_support_weights,
    )
    _require_nonnegative_floats(
        "--tracklet-max-track-support-rewards",
        args.tracklet_max_track_support_rewards,
    )


def _require_nonempty(name: str, values: list[object]) -> None:
    if not values:
        raise SystemExit(f"{name} must contain at least one value")


def _require_positive_ints(name: str, values: list[int]) -> None:
    _require_nonempty(name, values)
    if any(int(value) < 1 for value in values):
        raise SystemExit(f"{name} values must be >= 1")


def _require_nonnegative_floats(name: str, values: list[float]) -> None:
    _require_nonempty(name, values)
    if any(float(value) < 0.0 for value in values):
        raise SystemExit(f"{name} values must be >= 0")


if __name__ == "__main__":
    raise SystemExit(main())
