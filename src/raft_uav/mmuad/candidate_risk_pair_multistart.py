"""Compose risk-adjusted reservoirs, pair-state priors, and multi-start MAP.

The MMUAD candidate-assignment pipeline has three distinct failure modes:

* a good candidate can be removed by a small score-only reservoir;
* framewise scores can prefer a geometrically plausible but acceleration-inconsistent
  distractor sequence;
* alternating candidate-mixture MAP can converge to a poor local solution.

This module addresses the three stages without using validation/test truth for
selection.  Candidate uncertainty is used for branch-preserving reservoir
selection, an exact pair-state forward-backward model attaches an acceleration-
aware soft prior, and branch-seeded multi-start robust mixture-MAP estimates the
trajectory.  Optional truth is used only for diagnostics.
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
from raft_uav.mmuad.candidate_pair_forward_backward import (
    DEFAULT_OUTPUT_SCORE_COLUMN as PAIR_SCORE_COLUMN,
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
    pair_forward_backward_summary,
)
from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_oracle_recall_tables,
    load_candidate_inputs,
)
from raft_uav.mmuad.candidate_risk_reservoir import (
    DEFAULT_OUTPUT_SCORE_COLUMN as RISK_SCORE_COLUMN,
    RISK_SCORE_MODES,
    build_risk_adjusted_reservoir,
    risk_adjusted_reservoir_summary,
)
from raft_uav.mmuad.candidate_risk_reservoir_multistart import CandidateRiskScoreConfig
from raft_uav.mmuad.io import load_truth_csv
from raft_uav.mmuad.submission import load_sequence_class_map

SCORED_CANDIDATES_CSV = "mmuad_risk_pair_multistart_scored_candidates.csv"
RESERVOIR_CANDIDATES_CSV = "mmuad_risk_pair_multistart_reservoir.csv"
PAIR_CANDIDATES_CSV = "mmuad_risk_pair_multistart_pair_candidates.csv"
ORACLE_FRAMES_CSV = "mmuad_risk_pair_multistart_oracle_frames.csv"
ORACLE_SUMMARY_CSV = "mmuad_risk_pair_multistart_oracle_summary.csv"
ORACLE_BY_SEQUENCE_CSV = "mmuad_risk_pair_multistart_oracle_by_sequence.csv"
COMBINED_SUMMARY_JSON = "mmuad_risk_pair_multistart_summary.json"


@dataclass(frozen=True)
class CandidateRiskPairMultiStartResult:
    """Intermediate candidate tables, selected trajectory, and diagnostics."""

    scored_candidates: pd.DataFrame
    reservoir: pd.DataFrame
    pair_candidates: pd.DataFrame
    multistart: CandidateMixtureMultiStartResult
    oracle_frames: pd.DataFrame
    oracle_summary: pd.DataFrame
    oracle_by_sequence: pd.DataFrame
    summary: dict[str, Any]


def run_risk_pair_multistart(
    candidates: pd.DataFrame,
    *,
    risk_config: CandidateRiskScoreConfig | None = None,
    reservoir_config: ReservoirConfig | None = None,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    multistart_config: CandidateMixtureMultiStartConfig | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
    oracle_top_k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
    max_truth_time_delta_s: float = 0.5,
) -> CandidateRiskPairMultiStartResult:
    """Run the full truth-free candidate-assignment stack."""

    risk_config = risk_config or CandidateRiskScoreConfig()
    reservoir_config = reservoir_config or ReservoirConfig(
        score_column=risk_config.output_score_column
    )
    pair_config = pair_config or CandidatePairForwardBackwardConfig(
        score_column=risk_config.score_column,
        fallback_score_columns=(
            risk_config.output_score_column,
            risk_config.fallback_score_column,
            "confidence",
        ),
        sigma_column=risk_config.sigma_column,
    )
    mixture_config = mixture_config or core.CandidateMixtureMapConfig(
        top_k=0,
        score_column=pair_config.output_score_column,
        fallback_score_columns=(pair_config.score_column, *pair_config.fallback_score_columns),
        sigma_column=pair_config.sigma_column,
        score_normalization="none",
        # The pair posterior already includes the unary uncertainty prior.  Sigma is
        # still used to scale geometric residuals and measurement precision, but the
        # extra log-sigma prior is disabled by default to avoid double counting.
        sigma_log_weight=0.0,
    )
    mixture_config = core.CandidateMixtureMapConfig(
        **{
            **asdict(mixture_config),
            "top_k": 0,
            "score_column": pair_config.output_score_column,
            "fallback_score_columns": (
                pair_config.score_column,
                *pair_config.fallback_score_columns,
            ),
            "score_normalization": "none",
        }
    )
    multistart_config = multistart_config or CandidateMixtureMultiStartConfig()

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
    pair_candidates = attach_pair_forward_backward_candidate_prior(
        reservoir,
        config=pair_config,
    )
    multistart = run_multistart_candidate_mixture_map(
        pair_candidates.rows,
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
        "schema": "raft-uav-mmuad-risk-pair-multistart-v1",
        "risk_config": asdict(risk_config),
        "reservoir_config": asdict(reservoir_config),
        "pair_config": asdict(pair_config),
        "mixture_config": asdict(mixture_config),
        "multistart_config": asdict(multistart_config),
        "risk_reservoir": risk_adjusted_reservoir_summary(
            scored,
            reservoir,
            output_score_column=risk_config.output_score_column,
        ),
        "pair_prior": pair_forward_backward_summary(
            pair_candidates,
            score_column=pair_config.output_score_column,
        ),
        "selected_start": multistart.selected_start,
        "multistart": multistart.summary,
        "truth_used_for_selection": False,
        "mixture_uses_pair_posterior": True,
        "reservoir_oracle": {
            "top_k_values": [int(value) for value in oracle_top_k_values],
            "max_truth_time_delta_s": float(max_truth_time_delta_s),
            "frame_count": int(len(oracle_frames)),
            "pooled": _first_record(oracle_summary),
        }
        if truth is not None
        else None,
    }
    return CandidateRiskPairMultiStartResult(
        scored_candidates=scored.rows,
        reservoir=reservoir.rows,
        pair_candidates=pair_candidates.rows,
        multistart=multistart,
        oracle_frames=oracle_frames,
        oracle_summary=oracle_summary,
        oracle_by_sequence=oracle_by_sequence,
        summary=_jsonable(summary),
    )


def write_risk_pair_multistart_outputs(
    result: CandidateRiskPairMultiStartResult,
    *,
    output_dir: Path,
    class_map: Mapping[str, str] | None = None,
    default_classification: int | str = 2,
    official_results_csv: Path | None = None,
    official_zip: Path | None = None,
) -> dict[str, Path]:
    """Write the selected trajectory, intermediate candidate tables, and provenance."""

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
    table_paths = {
        "scored_candidates_csv": output / SCORED_CANDIDATES_CSV,
        "reservoir_candidates_csv": output / RESERVOIR_CANDIDATES_CSV,
        "pair_candidates_csv": output / PAIR_CANDIDATES_CSV,
    }
    result.scored_candidates.to_csv(table_paths["scored_candidates_csv"], index=False)
    result.reservoir.to_csv(table_paths["reservoir_candidates_csv"], index=False)
    result.pair_candidates.to_csv(table_paths["pair_candidates_csv"], index=False)
    paths.update(table_paths)
    if not result.oracle_frames.empty:
        oracle_paths = {
            "oracle_frames_csv": output / ORACLE_FRAMES_CSV,
            "oracle_summary_csv": output / ORACLE_SUMMARY_CSV,
            "oracle_by_sequence_csv": output / ORACLE_BY_SEQUENCE_CSV,
        }
        result.oracle_frames.to_csv(oracle_paths["oracle_frames_csv"], index=False)
        result.oracle_summary.to_csv(oracle_paths["oracle_summary_csv"], index=False)
        result.oracle_by_sequence.to_csv(oracle_paths["oracle_by_sequence_csv"], index=False)
        paths.update(oracle_paths)
    summary_path = output / COMBINED_SUMMARY_JSON
    summary_path.write_text(json.dumps(_jsonable(result.summary), indent=2), encoding="utf-8")
    paths["combined_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-risk-pair-multistart",
        description=(
            "run uncertainty-aware reservoir selection, pair-state candidate priors, "
            "and branch-seeded robust mixture-MAP"
        ),
    )
    parser.add_argument("--candidate-csv", action="append", default=[], metavar="BRANCH=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--external-initial-estimates-csv", type=Path)

    parser.add_argument("--risk-score-column", default="candidate_class_calibrated_score")
    parser.add_argument("--risk-fallback-score-column", default="ranker_score")
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--risk-output-score-column", default=RISK_SCORE_COLUMN)
    parser.add_argument("--risk-mode", choices=RISK_SCORE_MODES, default="logit-minus-log-sigma")
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--sigma-floor-m", type=float, default=1.0)

    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--cap-reason-bonus", type=float, default=0.0)

    parser.add_argument("--pair-score-column", default="candidate_class_calibrated_score")
    parser.add_argument("--pair-output-score-column", default=PAIR_SCORE_COLUMN)
    parser.add_argument("--pair-score-weight", type=float, default=1.0)
    parser.add_argument("--pair-sigma-log-weight", type=float, default=1.0)
    parser.add_argument("--transition-distance-std-m", type=float, default=2.0)
    parser.add_argument("--transition-speed-std-mps", type=float, default=15.0)
    parser.add_argument("--max-speed-mps", type=float, default=80.0)
    parser.add_argument("--speed-gate-penalty", type=float, default=25.0)
    parser.add_argument("--acceleration-std-mps2", type=float, default=20.0)
    parser.add_argument("--max-acceleration-mps2", type=float, default=80.0)
    parser.add_argument("--acceleration-gate-penalty", type=float, default=25.0)
    parser.add_argument("--source-switch-penalty", type=float, default=0.25)
    parser.add_argument("--branch-switch-penalty", type=float, default=0.25)
    parser.add_argument("--track-continuation-bonus", type=float, default=0.5)

    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument("--mixture-score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--mixture-sigma-log-weight", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)

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
    pair_config = CandidatePairForwardBackwardConfig(
        score_column=str(args.pair_score_column),
        fallback_score_columns=(
            str(args.risk_output_score_column),
            str(args.risk_fallback_score_column),
            "confidence",
        ),
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_weight=float(args.pair_score_weight),
        sigma_log_weight=float(args.pair_sigma_log_weight),
        transition_distance_std_m=float(args.transition_distance_std_m),
        transition_speed_std_mps=float(args.transition_speed_std_mps),
        max_speed_mps=float(args.max_speed_mps),
        speed_gate_penalty=float(args.speed_gate_penalty),
        acceleration_std_mps2=float(args.acceleration_std_mps2),
        max_acceleration_mps2=float(args.max_acceleration_mps2),
        acceleration_gate_penalty=float(args.acceleration_gate_penalty),
        source_switch_penalty=float(args.source_switch_penalty),
        branch_switch_penalty=float(args.branch_switch_penalty),
        track_continuation_bonus=float(args.track_continuation_bonus),
        output_score_column=str(args.pair_output_score_column),
    )
    mixture_config = core.CandidateMixtureMapConfig(
        top_k=0,
        score_column=str(args.pair_output_score_column),
        fallback_score_columns=(str(args.pair_score_column), str(args.risk_output_score_column)),
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_normalization="none",
        score_weight=float(args.mixture_score_weight),
        temperature=float(args.temperature),
        sigma_log_weight=float(args.mixture_sigma_log_weight),
        loss="huber",
        huber_delta=float(args.huber_delta),
        smoothness_weight=float(args.smoothness_weight),
        iterations=int(args.iterations),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
    )
    multistart_config = CandidateMixtureMultiStartConfig(
        include_score_top1=not bool(args.no_score_top1_start),
        include_frame_median=not bool(args.no_frame_median_start),
        include_branch_starts=not bool(args.no_branch_starts),
        max_branch_starts=int(args.max_branch_starts),
        min_branch_frame_fraction=float(args.min_branch_frame_fraction),
    )
    top_k_values = tuple(args.oracle_top_k) if args.oracle_top_k else (1, 3, 5, 10, 20)
    result = run_risk_pair_multistart(
        candidates,
        risk_config=risk_config,
        reservoir_config=reservoir_config,
        pair_config=pair_config,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external,
        truth=truth,
        oracle_top_k_values=top_k_values,
        max_truth_time_delta_s=float(args.max_truth_time_delta_s),
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_risk_pair_multistart_outputs(
        result,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        official_results_csv=args.official_results_csv,
        official_zip=args.official_zip,
    )
    print("mmuad_risk_pair_multistart=ok")
    print(f"selected_start={result.multistart.selected_start}")
    print(f"scored_candidate_rows={len(result.scored_candidates)}")
    print(f"reservoir_rows={len(result.reservoir)}")
    print(f"pair_candidate_rows={len(result.pair_candidates)}")
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
