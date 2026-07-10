"""Run risk-adjusted reservoir selection followed by multi-start mixture-MAP.

The learned candidate sigma helps robust mixture-MAP, but it can only help candidates
that survive the bounded reservoir.  Risk-adjusted reservoir scoring moves uncertainty
into that earlier selection step, while branch-seeded multi-start inference reduces the
chance that the alternating mixture solver settles on a poor local assignment.

This module deliberately keeps the two score uses separate by default:

* reservoir selection uses the risk-adjusted score;
* mixture-MAP uses the original calibrated/ranker score plus learned sigma.

That avoids counting sigma twice unless an experiment explicitly requests the risk score
as the mixture score as well.  Truth is optional and is used only for diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.candidate_mixture_map_multistart import (
    CandidateMixtureMultiStartConfig,
    CandidateMixtureMultiStartResult,
    run_multistart_candidate_mixture_map,
    write_multistart_candidate_mixture_outputs,
)
from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_oracle_recall_tables,
    load_candidate_inputs,
)
from raft_uav.mmuad.candidate_risk_reservoir import (
    DEFAULT_OUTPUT_SCORE_COLUMN,
    RISK_SCORE_MODES,
    build_risk_adjusted_reservoir,
    risk_adjusted_reservoir_summary,
)
from raft_uav.mmuad.io import load_truth_csv
from raft_uav.mmuad.submission import load_sequence_class_map

SCORED_CANDIDATES_CSV = "mmuad_risk_reservoir_multistart_scored_candidates.csv"
RESERVOIR_CANDIDATES_CSV = "mmuad_risk_reservoir_multistart_reservoir.csv"
ORACLE_FRAMES_CSV = "mmuad_risk_reservoir_multistart_oracle_frames.csv"
ORACLE_SUMMARY_CSV = "mmuad_risk_reservoir_multistart_oracle_summary.csv"
ORACLE_BY_SEQUENCE_CSV = "mmuad_risk_reservoir_multistart_oracle_by_sequence.csv"
COMBINED_SUMMARY_JSON = "mmuad_risk_reservoir_multistart_summary.json"


@dataclass(frozen=True)
class CandidateRiskScoreConfig:
    """Configuration for inference-safe risk-adjusted reservoir scoring."""

    score_column: str = "candidate_class_calibrated_score"
    fallback_score_column: str = "ranker_score"
    sigma_column: str = "predicted_sigma_m"
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN
    mode: str = "logit-minus-log-sigma"
    uncertainty_weight: float = 1.0
    sigma_floor_m: float = 1.0


@dataclass(frozen=True)
class CandidateRiskReservoirMultiStartResult:
    """Intermediate candidate tables, selected trajectory, and diagnostics."""

    scored_candidates: pd.DataFrame
    reservoir: pd.DataFrame
    multistart: CandidateMixtureMultiStartResult
    oracle_frames: pd.DataFrame
    oracle_summary: pd.DataFrame
    oracle_by_sequence: pd.DataFrame
    summary: dict[str, Any]


def run_risk_reservoir_multistart(
    candidates: pd.DataFrame,
    *,
    risk_config: CandidateRiskScoreConfig | None = None,
    reservoir_config: ReservoirConfig | None = None,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    multistart_config: CandidateMixtureMultiStartConfig | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    oracle_top_k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    max_truth_time_delta_s: float = 0.5,
) -> CandidateRiskReservoirMultiStartResult:
    """Run uncertainty-aware reservoir selection and truth-free multi-start inference."""

    risk_config = risk_config or CandidateRiskScoreConfig()
    reservoir_config = reservoir_config or ReservoirConfig(
        score_column=risk_config.output_score_column
    )
    multistart_config = multistart_config or CandidateMixtureMultiStartConfig()
    if mixture_config is None:
        mixture_config = core.CandidateMixtureMapConfig(
            top_k=0,
            score_column=risk_config.score_column,
            fallback_score_columns=(risk_config.fallback_score_column, "confidence"),
            sigma_column=risk_config.sigma_column,
        )
    else:
        mixture_config = core.CandidateMixtureMapConfig(
            **{
                **asdict(mixture_config),
                # The reservoir already bounds each frame.  A second global top-K
                # would undo branch/source preservation before multi-start inference.
                "top_k": 0,
            }
        )

    scored, reservoir = build_risk_adjusted_reservoir(
        candidates,
        score_column=risk_config.score_column,
        fallback_score_column=risk_config.fallback_score_column,
        sigma_column=risk_config.sigma_column,
        output_score_column=risk_config.output_score_column,
        mode=risk_config.mode,
        uncertainty_weight=risk_config.uncertainty_weight,
        sigma_floor_m=risk_config.sigma_floor_m,
        reservoir_config=reservoir_config,
    )
    multistart = run_multistart_candidate_mixture_map(
        reservoir.rows,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external_initial_estimates,
        truth=truth,
    )

    oracle_frames = pd.DataFrame()
    oracle_summary = pd.DataFrame()
    oracle_by_sequence = pd.DataFrame()
    if truth is not None:
        oracle_frames, oracle_summary, oracle_by_sequence = build_oracle_recall_tables(
            reservoir.rows,
            truth,
            top_k_values=tuple(oracle_top_k_values),
            max_truth_time_delta_s=float(max_truth_time_delta_s),
        )

    summary = {
        "schema": "raft-uav-mmuad-risk-reservoir-multistart-v1",
        "risk_config": asdict(risk_config),
        "reservoir_config": asdict(reservoir_config),
        "mixture_config": asdict(mixture_config),
        "multistart_config": asdict(multistart_config),
        "risk_reservoir": risk_adjusted_reservoir_summary(
            scored,
            reservoir,
            output_score_column=risk_config.output_score_column,
        ),
        "selected_start": multistart.selected_start,
        "multistart": multistart.summary,
        "truth_used_for_selection": False,
        "mixture_uses_risk_adjusted_score": (
            mixture_config.score_column == risk_config.output_score_column
        ),
        "reservoir_oracle": {
            "top_k_values": [int(value) for value in oracle_top_k_values],
            "max_truth_time_delta_s": float(max_truth_time_delta_s),
            "frame_count": int(len(oracle_frames)),
            "pooled": _first_record(oracle_summary),
        }
        if truth is not None
        else None,
    }
    return CandidateRiskReservoirMultiStartResult(
        scored_candidates=scored.rows,
        reservoir=reservoir.rows,
        multistart=multistart,
        oracle_frames=oracle_frames,
        oracle_summary=oracle_summary,
        oracle_by_sequence=oracle_by_sequence,
        summary=_jsonable(summary),
    )


def write_risk_reservoir_multistart_outputs(
    result: CandidateRiskReservoirMultiStartResult,
    *,
    output_dir: Path,
    class_map: Mapping[str, str] | None = None,
    default_classification: int | str = 2,
    official_results_csv: Path | None = None,
    official_zip: Path | None = None,
) -> dict[str, Path]:
    """Write selected trajectory, intermediate pools, oracle rows, and provenance."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = write_multistart_candidate_mixture_outputs(
        result.multistart,
        output_dir=output,
        class_map=class_map,
        default_classification=default_classification,
        official_results_csv=official_results_csv,
        official_zip=official_zip,
    )
    scored_path = output / SCORED_CANDIDATES_CSV
    reservoir_path = output / RESERVOIR_CANDIDATES_CSV
    result.scored_candidates.to_csv(scored_path, index=False)
    result.reservoir.to_csv(reservoir_path, index=False)
    paths["scored_candidates_csv"] = scored_path
    paths["reservoir_candidates_csv"] = reservoir_path
    if not result.oracle_frames.empty:
        oracle_frames_path = output / ORACLE_FRAMES_CSV
        oracle_summary_path = output / ORACLE_SUMMARY_CSV
        oracle_by_sequence_path = output / ORACLE_BY_SEQUENCE_CSV
        result.oracle_frames.to_csv(oracle_frames_path, index=False)
        result.oracle_summary.to_csv(oracle_summary_path, index=False)
        result.oracle_by_sequence.to_csv(oracle_by_sequence_path, index=False)
        paths["oracle_frames_csv"] = oracle_frames_path
        paths["oracle_summary_csv"] = oracle_summary_path
        paths["oracle_by_sequence_csv"] = oracle_by_sequence_path
    summary_path = output / COMBINED_SUMMARY_JSON
    summary_path.write_text(json.dumps(_jsonable(result.summary), indent=2), encoding="utf-8")
    paths["combined_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-risk-reservoir-multistart",
        description="run risk-adjusted reservoir selection and branch-seeded mixture-MAP",
    )
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--external-initial-estimates-csv", type=Path)

    parser.add_argument("--risk-score-column", default="candidate_class_calibrated_score")
    parser.add_argument("--risk-fallback-score-column", default="ranker_score")
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--risk-output-score-column", default=DEFAULT_OUTPUT_SCORE_COLUMN)
    parser.add_argument("--risk-mode", choices=RISK_SCORE_MODES, default="logit-minus-log-sigma")
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--sigma-floor-m", type=float, default=1.0)

    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--cap-reason-bonus", type=float, default=0.0)

    parser.add_argument("--mixture-score-column", default="candidate_class_calibrated_score")
    parser.add_argument("--mixture-fallback-score-column", action="append", default=[])
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization",
        choices=core.SCORE_NORMALIZATION_CHOICES,
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=core.LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=core.INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )

    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--max-branch-starts", type=int, default=8)
    parser.add_argument("--min-branch-frame-fraction", type=float, default=0.05)
    parser.add_argument("--no-score-top1-start", action="store_true")
    parser.add_argument("--no-frame-median-start", action="store_true")
    parser.add_argument("--no-branch-starts", action="store_true")

    parser.add_argument("--oracle-top-k", type=int, action="append", default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    candidates = load_candidate_inputs(args.candidate_csv)
    truth = None if args.truth_csv is None else load_truth_csv(args.truth_csv).rows
    external = None
    if args.external_initial_estimates_csv is not None:
        external = pd.read_csv(
            args.external_initial_estimates_csv,
            dtype=str,
            keep_default_na=False,
        )
        external.columns = [str(column).strip() for column in external.columns]

    risk_config = CandidateRiskScoreConfig(
        score_column=str(args.risk_score_column),
        fallback_score_column=str(args.risk_fallback_score_column),
        sigma_column=str(args.sigma_column),
        output_score_column=str(args.risk_output_score_column),
        mode=str(args.risk_mode),
        uncertainty_weight=float(args.uncertainty_weight),
        sigma_floor_m=float(args.sigma_floor_m),
    )
    reservoir_config = ReservoirConfig(
        global_top_n=int(args.global_top_n),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        max_candidates_per_frame=int(args.max_candidates_per_frame),
        score_column=str(args.risk_output_score_column),
        fallback_score_column=str(args.risk_fallback_score_column),
        score_floor_quantile=args.score_floor_quantile,
        cap_reason_bonus=float(args.cap_reason_bonus),
    )
    fallback = tuple(args.mixture_fallback_score_column) or (
        str(args.risk_fallback_score_column),
        "confidence",
    )
    mixture_config = core.CandidateMixtureMapConfig(
        top_k=0,
        score_column=str(args.mixture_score_column),
        fallback_score_columns=fallback,
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_normalization=str(args.score_normalization),
        score_weight=float(args.score_weight),
        temperature=float(args.temperature),
        sigma_log_weight=float(args.sigma_log_weight),
        loss=str(args.loss),
        huber_delta=float(args.huber_delta),
        smoothness_weight=float(args.smoothness_weight),
        iterations=int(args.iterations),
        tolerance_m=float(args.tolerance_m),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
        initialization=str(args.initialization),
    )
    multistart_config = CandidateMixtureMultiStartConfig(
        branch_column=str(args.branch_column),
        include_score_top1=not bool(args.no_score_top1_start),
        include_frame_median=not bool(args.no_frame_median_start),
        include_branch_starts=not bool(args.no_branch_starts),
        max_branch_starts=int(args.max_branch_starts),
        min_branch_frame_fraction=float(args.min_branch_frame_fraction),
    )
    top_k_values = tuple(args.oracle_top_k) if args.oracle_top_k else (1, 3, 5, 10, 20)
    result = run_risk_reservoir_multistart(
        candidates,
        risk_config=risk_config,
        reservoir_config=reservoir_config,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external,
        truth=truth,
        oracle_top_k_values=top_k_values,
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_risk_reservoir_multistart_outputs(
        result,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        official_results_csv=args.official_results_csv,
        official_zip=args.official_zip,
    )
    print("mmuad_risk_reservoir_multistart=ok")
    print(f"selected_start={result.multistart.selected_start}")
    print(f"scored_candidate_rows={len(result.scored_candidates)}")
    print(f"reservoir_rows={len(result.reservoir)}")
    pooled = result.multistart.selected_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"diagnostic_rmse_3d_m={pooled['rmse_3d_m']}")
    if not result.oracle_summary.empty:
        print(f"reservoir_oracle_all_mse={result.oracle_summary.loc[0, 'oracle_all_3d_m_mse']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _first_record(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    return _jsonable(frame.iloc[0].to_dict())


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
