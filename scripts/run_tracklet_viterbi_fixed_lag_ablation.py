"""Run fixed-lag tracklet-Viterbi radar association ablations."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ablation_common as common  # noqa: E402
from raft_uav.tracklet_viterbi_fixed_lag_cli import (  # noqa: E402
    _FIXED_LAG_ENV,
    main as fixed_lag_tracklet_viterbi_main,
)


@dataclass(frozen=True)
class _Config:
    name: str
    threshold: float
    viterbi_lag_s: float


def main() -> int:
    parser = argparse.ArgumentParser()
    common.add_experiment_io_arguments(
        parser,
        default_output_dir=Path("outputs/tracklet_viterbi_fixed_lag_ablation"),
        default_summary_output=Path("outputs/tracklet_viterbi_fixed_lag_ablation.csv"),
    )
    parser.add_argument("--thresholds", nargs="*", type=float, default=[0.3, 0.4, 0.5])
    parser.add_argument("--viterbi-lags-s", nargs="*", type=float, default=[10.0, 20.0, 30.0])
    common.add_fixed_lag_argument(parser)
    common.add_soft_update_arguments(parser)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    configs = [
        _Config(_config_name(threshold, lag_s), threshold, lag_s)
        for threshold in args.thresholds
        for lag_s in args.viterbi_lags_s
    ]
    rows = common.run_named_config_experiments(args, configs, _run_one, _candidate_row)
    common.write_summary_csv(args.summary_output, rows)
    print(f"wrote {len(rows)} rows to {args.summary_output}")
    return 0


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
            "viterbi_lag_s": config.viterbi_lag_s,
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
        "tracklet-viterbi-fixed-lag",
        "--radar-catprob-threshold",
        str(config.threshold),
        *[str(option) for option in common.robust_update_options(args)],
        *[str(option) for option in common.smoother_options("fixed-lag", args.fixed_lag_s)],
    ]
    print(
        f"{_FIXED_LAG_ENV}={config.viterbi_lag_s} "
        + "python -m raft_uav.tracklet_viterbi_fixed_lag_cli "
        + " ".join(cli_args),
        flush=True,
    )
    with _temporary_env(_FIXED_LAG_ENV, str(config.viterbi_lag_s)):
        status = fixed_lag_tracklet_viterbi_main(cli_args)
    if status != 0:
        raise RuntimeError(f"fixed-lag tracklet-viterbi run failed with status {status}")


@contextmanager
def _temporary_env(name: str, value: str) -> Iterator[None]:
    old_value = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_value


def _config_name(threshold: float, lag_s: float) -> str:
    return (
        f"tracklet_viterbi_fixed_lag_t{common.slug(threshold, precision=2)}"
        f"_lag{common.slug(lag_s, precision=1)}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
