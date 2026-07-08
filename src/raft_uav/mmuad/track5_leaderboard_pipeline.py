"""End-to-end MMUAD Track 5 leaderboard packaging pipeline.

This module chains the current inference-safe CVPR/UG2+ Track 5 components:

1. branch-preserving candidate reservoir;
2. robust candidate-mixture MAP trajectory smoothing;
3. official-template resampling and ZIP preflight validation.

It exists to reduce leaderboard-submission drift: experiments can produce
candidate/sensor-time trajectories, while Codabench requires exactly the official
``Sequence,Timestamp`` template rows.  Optionally, the final template projection
can run through the acceleration-regularized Track 5 post-processor so the main
leaderboard command can produce a smoothed upload artifact without an extra
manual step.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import CandidateMixtureMapConfig
from raft_uav.mmuad.candidate_reservoir import ReservoirConfig
from raft_uav.mmuad.candidate_reservoir import load_candidate_inputs
from raft_uav.mmuad.candidate_reservoir_mixture_map import run_reservoir_mixture_map
from raft_uav.mmuad.candidate_reservoir_mixture_map import write_reservoir_mixture_map_outputs
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.submission import load_official_track5_template_file, load_sequence_class_map
from raft_uav.mmuad.track5_template_resample import RESAMPLE_METHODS
from raft_uav.mmuad.track5_template_resample import ResampleMethod
from raft_uav.mmuad.track5_template_resample import write_track5_template_resample_outputs
from raft_uav.mmuad.track5_trajectory_regularizer import run_track5_trajectory_regularizer

PIPELINE_MANIFEST_JSON = "mmuad_track5_leaderboard_pipeline_manifest.json"
MIXTURE_DIR = "reservoir_mixture"
SUBMISSION_DIR = "track5_submission"


def run_track5_leaderboard_pipeline(
    *,
    candidates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    reservoir_config: ReservoirConfig | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    class_map: dict[str, str] | None = None,
    default_classification: int | str = 2,
    max_template_nearest_time_delta_s: float | None = None,
    submission_resample_method: ResampleMethod = "linear",
    submission_max_interpolation_gap_s: float | None = None,
    apply_final_regularizer: bool = False,
    regularizer_smoothness_weight: float = 10.0,
    regularizer_huber_delta_m: float = 25.0,
    regularizer_iterations: int = 5,
    regularizer_observation_sigma_m: float = 10.0,
) -> dict[str, Any]:
    """Run reservoir mixture-MAP and package it against a Track 5 template.

    ``submission_resample_method`` and ``submission_max_interpolation_gap_s``
    control the final Codabench-template projection.  Exposing them here avoids
    an error-prone extra post-processing step when dense sensor-time trajectories
    need nearest-only or large-gap-safe template packaging.

    When ``apply_final_regularizer`` is true, the package step uses the Track 5
    acceleration-regularized robust smoother instead of plain template resampling.
    This keeps final-stage smoothing in the same provenance manifest as the
    reservoir/mixture run and avoids submitting an unsmoothed intermediate by
    mistake.
    """

    output = Path(output_dir)
    mixture_dir = output / MIXTURE_DIR
    submission_dir = output / SUBMISSION_DIR
    reservoir, mixture_result, mixture_summary = run_reservoir_mixture_map(
        candidates,
        reservoir_config=reservoir_config,
        mixture_config=mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    mixture_paths = write_reservoir_mixture_map_outputs(
        reservoir=reservoir,
        result=mixture_result,
        summary=mixture_summary,
        output_dir=mixture_dir,
        class_map=class_map,
        default_classification=default_classification,
    )
    if apply_final_regularizer:
        submission_paths = run_track5_trajectory_regularizer(
            estimates=mixture_result.estimates,
            template=template,
            output_dir=submission_dir,
            class_map=class_map,
            default_classification=default_classification,
            max_nearest_time_delta_s=max_template_nearest_time_delta_s,
            resample_method=submission_resample_method,
            max_interpolation_gap_s=submission_max_interpolation_gap_s,
            smoothness_weight=float(regularizer_smoothness_weight),
            huber_delta_m=float(regularizer_huber_delta_m),
            iterations=int(regularizer_iterations),
            observation_sigma_m=float(regularizer_observation_sigma_m),
        )
    else:
        submission_paths = write_track5_template_resample_outputs(
            estimates=mixture_result.estimates,
            template=template,
            output_dir=submission_dir,
            class_map=class_map,
            default_classification=default_classification,
            max_nearest_time_delta_s=max_template_nearest_time_delta_s,
            resample_method=submission_resample_method,
            max_interpolation_gap_s=submission_max_interpolation_gap_s,
        )
    manifest = {
        "schema": "raft-uav-mmuad-track5-leaderboard-pipeline-v3",
        "reservoir_config": asdict(reservoir_config or ReservoirConfig()),
        "mixture_config": asdict(
            _with_reservoir_top_k_zero(mixture_config or CandidateMixtureMapConfig()),
        ),
        "candidate_rows": int(len(candidates)),
        "reservoir_rows": int(len(reservoir)),
        "mixture_estimate_rows": int(len(mixture_result.estimates)),
        "template_row_count": int(len(template)),
        "default_classification": str(default_classification),
        "max_template_nearest_time_delta_s": max_template_nearest_time_delta_s,
        "submission_resample_method": str(submission_resample_method),
        "submission_max_interpolation_gap_s": submission_max_interpolation_gap_s,
        "final_regularizer_enabled": bool(apply_final_regularizer),
        "regularizer_smoothness_weight": float(regularizer_smoothness_weight),
        "regularizer_huber_delta_m": float(regularizer_huber_delta_m),
        "regularizer_iterations": int(regularizer_iterations),
        "regularizer_observation_sigma_m": float(regularizer_observation_sigma_m),
        "mixture_summary": mixture_summary,
        "mixture_paths": {name: str(path) for name, path in mixture_paths.items()},
        "submission_paths": {name: str(path) for name, path in submission_paths.items()},
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / PIPELINE_MANIFEST_JSON
    manifest_path.write_text(json.dumps(_jsonable(manifest), indent=2), encoding="utf-8")
    return {
        "reservoir": reservoir,
        "mixture_result": mixture_result,
        "mixture_paths": mixture_paths,
        "submission_paths": submission_paths,
        "manifest": manifest,
        "manifest_path": manifest_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-track5-leaderboard-pipeline",
        description="run reservoir mixture-MAP and package an official Track 5 ZIP",
    )
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses file stem as branch",
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--require-leaderboard-ready", action="store_true")
    parser.add_argument("--max-template-nearest-time-delta-s", type=float)
    parser.add_argument(
        "--submission-resample-method",
        choices=RESAMPLE_METHODS,
        default="linear",
        help="coordinate resampling mode for official-template submission rows",
    )
    parser.add_argument(
        "--submission-max-interpolation-gap-s",
        type=float,
        help="fallback to nearest when template interpolation spans a larger source-time gap",
    )
    parser.add_argument(
        "--final-regularizer",
        action="store_true",
        help="run the acceleration-regularized Track 5 post-processor before packaging",
    )
    parser.add_argument("--regularizer-smoothness-weight", type=float, default=10.0)
    parser.add_argument("--regularizer-huber-delta-m", type=float, default=25.0)
    parser.add_argument("--regularizer-iterations", type=int, default=5)
    parser.add_argument("--regularizer-observation-sigma-m", type=float, default=10.0)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--reservoir-score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--reservoir-fallback-score-column", default="confidence")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--mixture-score-column", default="candidate_reservoir_score")
    parser.add_argument("--mixture-fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    candidates = load_candidate_inputs(args.candidate_csv)
    template = load_official_track5_template_file(args.template)
    truth = None if args.truth_csv is None else load_evaluation_truth_file(args.truth_csv).rows
    initial_estimates = None
    if args.initial_estimates_csv is not None:
        initial_estimates = pd.read_csv(args.initial_estimates_csv)
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    reservoir_config = ReservoirConfig(
        global_top_n=int(args.global_top_n),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        max_candidates_per_frame=int(args.max_candidates_per_frame),
        score_column=str(args.reservoir_score_column),
        fallback_score_column=str(args.reservoir_fallback_score_column),
        score_floor_quantile=args.score_floor_quantile,
    )
    mixture_config = CandidateMixtureMapConfig(
        top_k=0,
        score_column=str(args.mixture_score_column),
        fallback_score_columns=tuple(args.mixture_fallback_score_column)
        or ("ranker_score", "confidence"),
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_weight=float(args.score_weight),
        temperature=float(args.temperature),
        sigma_log_weight=float(args.sigma_log_weight),
        loss="huber",
        huber_delta=float(args.huber_delta),
        smoothness_weight=float(args.smoothness_weight),
        iterations=int(args.iterations),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
    )
    result = run_track5_leaderboard_pipeline(
        candidates=candidates,
        template=template,
        output_dir=args.output_dir,
        reservoir_config=reservoir_config,
        mixture_config=mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
        class_map=class_map,
        default_classification=args.default_classification,
        max_template_nearest_time_delta_s=args.max_template_nearest_time_delta_s,
        submission_resample_method=args.submission_resample_method,
        submission_max_interpolation_gap_s=args.submission_max_interpolation_gap_s,
        apply_final_regularizer=bool(args.final_regularizer),
        regularizer_smoothness_weight=float(args.regularizer_smoothness_weight),
        regularizer_huber_delta_m=float(args.regularizer_huber_delta_m),
        regularizer_iterations=int(args.regularizer_iterations),
        regularizer_observation_sigma_m=float(args.regularizer_observation_sigma_m),
    )
    validation_json = result["submission_paths"]["validation_json"]
    validation_summary = json.loads(Path(validation_json).read_text(encoding="utf-8"))
    print("mmuad_track5_leaderboard_pipeline=ok")
    print(f"manifest_json={result['manifest_path']}")
    print(f"ug2_submission_zip={result['submission_paths']['official_zip']}")
    print(f"final_regularizer_enabled={result['manifest']['final_regularizer_enabled']}")
    print(f"leaderboard_ready={validation_summary.get('leaderboard_ready')}")
    print(f"codabench_upload_ready={validation_summary.get('codabench_upload_ready')}")
    if args.require_leaderboard_ready and not validation_summary.get("leaderboard_ready", False):
        reasons = ", ".join(validation_summary.get("leaderboard_blocking_reasons", [])) or "unknown"
        raise SystemExit(f"Track 5 pipeline output is not leaderboard-ready: {reasons}")
    return 0


def _with_reservoir_top_k_zero(config: CandidateMixtureMapConfig) -> CandidateMixtureMapConfig:
    return CandidateMixtureMapConfig(**{**asdict(config), "top_k": 0})


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
