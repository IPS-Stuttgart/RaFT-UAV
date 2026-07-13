"""Origin-group multiplicity correction for MMUAD pair-state priors.

Branch-preserving MMUAD reservoirs intentionally keep raw, calibrated, dynamic,
and merged hypotheses.  Several rows can therefore represent the same physical
cluster.  A pair-state forward-backward model sums over candidate paths, so
duplicated siblings can acquire excess posterior mass simply because more
equivalent paths exist.

This module subtracts ``correction_strength * log(group_size)`` from each
candidate emission before pair-state inference.  At strength one, identical
siblings collectively receive the same emission mass as a singleton group.
Candidate coordinates and branches remain separate for the downstream robust
mixture-MAP smoother.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    run_candidate_mixture_map,
    write_candidate_mixture_map_outputs,
)
from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    attach_pair_forward_backward_candidate_prior,
    pair_forward_backward_summary,
)
from raft_uav.mmuad.candidate_pair_forward_backward_agreement_adaptive import (
    AgreementAdaptivePairBlendConfig,
    agreement_adaptive_pair_summary,
    attach_agreement_adaptive_pair_prior,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


DEFAULT_GROUP_COLUMN_ALIASES = (
    "mmuad_calibration_origin_row",
    "candidate_origin_row",
    "origin_row",
)
MISSING_GROUP_POLICIES = ("unique", "error")
DEFAULT_CORRECTED_SCORE_COLUMN = "candidate_pair_group_corrected_emission_score"


@dataclass(frozen=True)
class PairGroupMultiplicityConfig:
    """Configuration for origin-group emission-mass correction."""

    group_column: str | None = None
    group_column_aliases: tuple[str, ...] = DEFAULT_GROUP_COLUMN_ALIASES
    correction_strength: float = 1.0
    missing_group_policy: str = "unique"
    corrected_score_column: str = DEFAULT_CORRECTED_SCORE_COLUMN


def prepare_group_corrected_pair_candidates(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    group_config: PairGroupMultiplicityConfig | None = None,
) -> tuple[pd.DataFrame, CandidatePairForwardBackwardConfig, dict[str, Any]]:
    """Attach group-corrected emissions and return an effective pair config."""

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    group_cfg = group_config or PairGroupMultiplicityConfig()
    _validate_group_config(group_cfg)
    if float(pair_cfg.score_weight) <= 0.0 and float(group_cfg.correction_strength) > 0.0:
        raise ValueError(
            "pair score_weight must be positive when group correction is enabled"
        )

    rows = _candidate_rows(candidates)
    resolved_group_column = _resolve_group_column(rows, group_cfg)
    if rows.empty:
        for column, dtype in (
            ("candidate_pair_hypothesis_group", str),
            ("candidate_pair_hypothesis_group_size", int),
            ("candidate_pair_group_base_normalized_score", float),
            ("candidate_pair_group_log_size_penalty", float),
            (group_cfg.corrected_score_column, float),
        ):
            rows[column] = pd.Series(dtype=dtype)
        effective = replace(
            pair_cfg,
            score_column=group_cfg.corrected_score_column,
            fallback_score_columns=(),
            score_normalization="none",
        )
        return rows, effective, _grouping_summary(
            rows,
            resolved_group_column=resolved_group_column,
            pair_config=pair_cfg,
            effective_config=effective,
            group_config=group_cfg,
        )

    rows = rows.reset_index(drop=True)
    rows["candidate_pair_input_row"] = np.arange(len(rows), dtype=int)
    rows["candidate_pair_hypothesis_group"] = _group_values(
        rows,
        resolved_group_column=resolved_group_column,
        missing_group_policy=group_cfg.missing_group_policy,
    )

    raw_score = _candidate_scores(rows, pair_cfg)
    normalized_score = pd.Series(np.nan, index=rows.index, dtype=float)
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        normalized_score.loc[frame.index] = _normalize_scores(
            raw_score.loc[frame.index].to_numpy(float),
            pair_cfg.score_normalization,
        )
    rows["candidate_pair_group_base_normalized_score"] = normalized_score

    group_size = rows.groupby(
        ["sequence_id", "time_s", "candidate_pair_hypothesis_group"],
        sort=False,
        dropna=False,
    )["candidate_pair_input_row"].transform("size")
    rows["candidate_pair_hypothesis_group_size"] = group_size.astype(int)
    penalty = float(group_cfg.correction_strength) * np.log(
        np.maximum(group_size.to_numpy(float), 1.0)
    )
    rows["candidate_pair_group_log_size_penalty"] = penalty

    score_weight = float(pair_cfg.score_weight)
    if abs(score_weight) <= 1.0e-12:
        corrected_score = normalized_score.to_numpy(float)
    else:
        corrected_score = normalized_score.to_numpy(float) - penalty / score_weight
    rows[group_cfg.corrected_score_column] = corrected_score

    effective = replace(
        pair_cfg,
        score_column=group_cfg.corrected_score_column,
        fallback_score_columns=(),
        score_normalization="none",
    )
    summary = _grouping_summary(
        rows,
        resolved_group_column=resolved_group_column,
        pair_config=pair_cfg,
        effective_config=effective,
        group_config=group_cfg,
    )
    return rows, effective, summary


def attach_group_corrected_pair_prior(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    group_config: PairGroupMultiplicityConfig | None = None,
    blend_config: AgreementAdaptivePairBlendConfig | None = None,
) -> tuple[CandidateFrame, CandidatePairForwardBackwardConfig, dict[str, Any]]:
    """Attach a group-corrected pair posterior, optionally agreement-adapted."""

    prepared, effective_pair_config, grouping_summary = (
        prepare_group_corrected_pair_candidates(
            candidates,
            pair_config=pair_config,
            group_config=group_config,
        )
    )
    if blend_config is None:
        augmented = attach_pair_forward_backward_candidate_prior(
            prepared,
            config=effective_pair_config,
        )
    else:
        augmented = attach_agreement_adaptive_pair_prior(
            prepared,
            pair_config=effective_pair_config,
            blend_config=blend_config,
        )
    return augmented, effective_pair_config, grouping_summary


def group_corrected_pair_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_score_column: str,
    adaptive_score_column: str | None = None,
) -> dict[str, Any]:
    """Return pair-posterior and multiplicity diagnostics."""

    rows = _candidate_rows(candidates)
    group_sizes = _numeric_column(rows, "candidate_pair_hypothesis_group_size")
    penalty = _numeric_column(rows, "candidate_pair_group_log_size_penalty")
    frame_groups = (
        rows[
            ["sequence_id", "time_s", "candidate_pair_hypothesis_group"]
        ].drop_duplicates()
        if "candidate_pair_hypothesis_group" in rows.columns
        else pd.DataFrame()
    )
    summary: dict[str, Any] = {
        "row_count": int(len(rows)),
        "frame_count": int(
            rows[["sequence_id", "time_s"]].drop_duplicates().shape[0]
        )
        if not rows.empty
        else 0,
        "physical_group_count": int(len(frame_groups)),
        "duplicate_candidate_rows": int((group_sizes > 1).sum())
        if len(group_sizes)
        else 0,
        "group_size_mean": _safe_mean(group_sizes),
        "group_size_p95": _safe_quantile(group_sizes, 0.95),
        "group_size_max": _safe_max(group_sizes),
        "log_size_penalty_mean": _safe_mean(penalty),
        "pair_summary": pair_forward_backward_summary(
            rows,
            score_column=pair_score_column,
        ),
    }
    if adaptive_score_column is not None:
        summary["agreement_adaptive_summary"] = agreement_adaptive_pair_summary(
            rows,
            pair_score_column=pair_score_column,
            output_score_column=adaptive_score_column,
        )
    return summary


def write_group_corrected_pair_outputs(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    effective_pair_config: CandidatePairForwardBackwardConfig | None = None,
    group_config: PairGroupMultiplicityConfig | None = None,
    grouping_summary: Mapping[str, Any] | None = None,
    blend_config: AgreementAdaptivePairBlendConfig | None = None,
    extra_summary: Mapping[str, Any] | None = None,
) -> None:
    """Write corrected pair candidates and optional provenance."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if summary_json is None:
        return

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    effective_cfg = effective_pair_config or pair_cfg
    group_cfg = group_config or PairGroupMultiplicityConfig()
    adaptive_column = None if blend_config is None else blend_config.output_score_column
    payload: dict[str, Any] = {
        "schema": "raft-uav-mmuad-pair-group-multiplicity-v1",
        "pair_config": asdict(pair_cfg),
        "effective_pair_config": asdict(effective_cfg),
        "group_config": asdict(group_cfg),
        "blend_config": None if blend_config is None else asdict(blend_config),
        "output_csv": str(output_csv),
        "grouping_summary": dict(grouping_summary or {}),
        "summary": group_corrected_pair_summary(
            candidates,
            pair_score_column=effective_cfg.output_score_column,
            adaptive_score_column=adaptive_column,
        ),
        "truth_used_for_candidate_prior": False,
    }
    if extra_summary:
        payload.update(dict(extra_summary))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-pair-group-correction",
        description=(
            "correct duplicate origin-group path multiplicity before MMUAD "
            "pair-state forward-backward inference"
        ),
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--group-column")
    parser.add_argument("--group-column-alias", action="append", default=[])
    parser.add_argument(
        "--missing-group-policy",
        choices=MISSING_GROUP_POLICIES,
        default="unique",
    )
    parser.add_argument("--correction-strength", type=float, default=1.0)
    parser.add_argument("--corrected-score-column", default=DEFAULT_CORRECTED_SCORE_COLUMN)

    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization",
        choices=("minmax", "rank", "none"),
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=1.0)
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
    parser.add_argument("--time-gap-penalty", type=float, default=0.0)
    parser.add_argument(
        "--pair-score-column",
        default="candidate_pair_group_corrected_forward_backward_score",
    )

    parser.add_argument("--agreement-adaptive", action="store_true")
    parser.add_argument("--min-pair-weight", type=float, default=0.0)
    parser.add_argument("--max-pair-weight", type=float, default=1.0)
    parser.add_argument("--entropy-power", type=float, default=1.0)
    parser.add_argument("--agreement-power", type=float, default=1.0)
    parser.add_argument("--agreement-floor", type=float, default=0.0)
    parser.add_argument(
        "--adaptive-score-column",
        default="candidate_pair_group_corrected_agreement_adaptive_score",
    )
    parser.add_argument("--replace-confidence", action="store_true")

    parser.add_argument("--mixture-output-dir", type=Path)
    parser.add_argument("--mixture-truth-csv", type=Path)
    parser.add_argument("--mixture-initial-estimates-csv", type=Path)
    parser.add_argument("--mixture-top-k", type=int, default=20)
    parser.add_argument("--mixture-smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--mixture-huber-delta", type=float, default=1.0)
    parser.add_argument("--mixture-iterations", type=int, default=5)
    parser.add_argument("--mixture-temperature", type=float, default=1.0)
    parser.add_argument("--mixture-score-weight", type=float, default=1.0)
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or (
        "candidate_risk_adjusted_score",
        "candidate_forward_backward_score",
        "branch_consensus_rank_score",
        "candidate_temporal_consensus_score",
        "ranker_score",
        "confidence",
    )
    pair_cfg = CandidatePairForwardBackwardConfig(
        score_column=str(args.score_column),
        fallback_score_columns=fallback_columns,
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_normalization=str(args.score_normalization),
        score_weight=float(args.score_weight),
        sigma_log_weight=float(args.sigma_log_weight),
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
        time_gap_penalty=float(args.time_gap_penalty),
        output_score_column=str(args.pair_score_column),
    )
    aliases = (
        tuple(args.group_column_alias)
        if args.group_column_alias
        else DEFAULT_GROUP_COLUMN_ALIASES
    )
    group_cfg = PairGroupMultiplicityConfig(
        group_column=args.group_column,
        group_column_aliases=aliases,
        correction_strength=float(args.correction_strength),
        missing_group_policy=str(args.missing_group_policy),
        corrected_score_column=str(args.corrected_score_column),
    )
    blend_cfg = (
        AgreementAdaptivePairBlendConfig(
            min_pair_weight=float(args.min_pair_weight),
            max_pair_weight=float(args.max_pair_weight),
            entropy_power=float(args.entropy_power),
            agreement_power=float(args.agreement_power),
            agreement_floor=float(args.agreement_floor),
            output_score_column=str(args.adaptive_score_column),
        )
        if args.agreement_adaptive
        else None
    )

    augmented, effective_pair_cfg, grouping_summary = attach_group_corrected_pair_prior(
        load_candidate_file(args.candidate_csv),
        pair_config=pair_cfg,
        group_config=group_cfg,
        blend_config=blend_cfg,
    )
    output_score_column = (
        effective_pair_cfg.output_score_column
        if blend_cfg is None
        else blend_cfg.output_score_column
    )
    if args.replace_confidence and not augmented.rows.empty:
        rows = augmented.rows.copy()
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence"), errors="coerce")
        rows["confidence"] = pd.to_numeric(
            rows[output_score_column],
            errors="coerce",
        )
        augmented = CandidateFrame(normalize_candidate_columns(rows))

    extra_summary: dict[str, Any] = {
        "candidate_csv": str(args.candidate_csv),
        "agreement_adaptive": bool(args.agreement_adaptive),
        "replace_confidence": bool(args.replace_confidence),
        "downstream_score_column": output_score_column,
    }
    if args.mixture_output_dir is not None:
        truth = (
            None
            if args.mixture_truth_csv is None
            else load_evaluation_truth_file(args.mixture_truth_csv).rows
        )
        initial_estimates = (
            None
            if args.mixture_initial_estimates_csv is None
            else pd.read_csv(args.mixture_initial_estimates_csv)
        )
        mixture_result = run_candidate_mixture_map(
            augmented.rows,
            config=CandidateMixtureMapConfig(
                top_k=int(args.mixture_top_k),
                score_column=output_score_column,
                fallback_score_columns=(effective_pair_cfg.output_score_column,),
                sigma_column=effective_pair_cfg.sigma_column,
                default_sigma_m=effective_pair_cfg.default_sigma_m,
                sigma_min_m=effective_pair_cfg.sigma_min_m,
                sigma_max_m=effective_pair_cfg.sigma_max_m,
                score_normalization="none",
                score_weight=float(args.mixture_score_weight),
                temperature=float(args.mixture_temperature),
                loss="huber",
                huber_delta=float(args.mixture_huber_delta),
                smoothness_weight=float(args.mixture_smoothness_weight),
                iterations=int(args.mixture_iterations),
            ),
            initial_estimates=initial_estimates,
            truth=truth,
        )
        mixture_paths = write_candidate_mixture_map_outputs(
            mixture_result,
            args.mixture_output_dir,
        )
        extra_summary["mixture_output_dir"] = str(args.mixture_output_dir)
        extra_summary["mixture_paths"] = {
            name: str(path) for name, path in mixture_paths.items()
        }
        extra_summary["mixture_summary"] = mixture_result.summary

    write_group_corrected_pair_outputs(
        augmented,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        pair_config=pair_cfg,
        effective_pair_config=effective_pair_cfg,
        group_config=group_cfg,
        grouping_summary=grouping_summary,
        blend_config=blend_cfg,
        extra_summary=extra_summary,
    )
    print("mmuad_pair_group_multiplicity_correction=ok")
    print(f"output_csv={args.output_csv}")
    print(f"rows={len(augmented.rows)}")
    print(f"downstream_score_column={output_score_column}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    if args.mixture_output_dir is not None:
        print(f"mixture_output_dir={args.mixture_output_dir}")
    return 0


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = (
        candidates.rows.copy()
        if isinstance(candidates, CandidateFrame)
        else pd.DataFrame(candidates).copy()
    )
    rows = normalize_candidate_columns(rows)
    if rows.empty:
        return rows
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    for column in ("time_s", "x_m", "y_m", "z_m"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    finite = np.isfinite(
        rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)
    ).all(axis=1)
    return rows.loc[finite].copy().reset_index(drop=True)


def _validate_group_config(config: PairGroupMultiplicityConfig) -> None:
    if not np.isfinite(config.correction_strength) or config.correction_strength < 0.0:
        raise ValueError("correction_strength must be finite and non-negative")
    if config.missing_group_policy not in MISSING_GROUP_POLICIES:
        raise ValueError(
            f"unsupported missing_group_policy={config.missing_group_policy!r}"
        )
    if not str(config.corrected_score_column).strip():
        raise ValueError("corrected_score_column must be non-empty")


def _resolve_group_column(
    rows: pd.DataFrame,
    config: PairGroupMultiplicityConfig,
) -> str | None:
    if config.group_column is not None:
        if config.group_column not in rows.columns and not rows.empty:
            raise ValueError(f"group column {config.group_column!r} is missing")
        return config.group_column
    for column in config.group_column_aliases:
        if column in rows.columns:
            return column
    if config.missing_group_policy == "error" and not rows.empty:
        raise ValueError(
            "no hypothesis-group column found; tried "
            + ", ".join(config.group_column_aliases)
        )
    return None


def _group_values(
    rows: pd.DataFrame,
    *,
    resolved_group_column: str | None,
    missing_group_policy: str,
) -> pd.Series:
    unique = pd.Series(
        [f"__candidate_row_{index}" for index in rows.index],
        index=rows.index,
        dtype=str,
    )
    if resolved_group_column is None:
        return unique
    raw = rows[resolved_group_column]
    missing = raw.isna() | raw.astype(str).str.strip().isin(("", "nan", "None"))
    if missing.any() and missing_group_policy == "error":
        indices = rows.index[missing].tolist()[:10]
        raise ValueError(
            f"group column {resolved_group_column!r} has missing values at rows {indices}"
        )
    values = raw.astype(str).str.strip()
    return values.where(~missing, unique)


def _candidate_scores(
    rows: pd.DataFrame,
    config: CandidatePairForwardBackwardConfig,
) -> pd.Series:
    for column in (config.score_column, *config.fallback_score_columns):
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        finite = np.isfinite(values.to_numpy(float))
        if finite.any():
            minimum = float(values.loc[finite].min())
            return values.fillna(minimum).astype(float)
    return pd.Series(1.0, index=rows.index, dtype=float)


def _normalize_scores(values: np.ndarray, mode: str) -> np.ndarray:
    score = np.asarray(values, dtype=float)
    finite = np.isfinite(score)
    if not finite.any():
        return np.zeros_like(score, dtype=float)
    floor = float(np.min(score[finite]))
    score = np.where(finite, score, floor)
    if mode == "none":
        return score
    if mode == "rank":
        if len(score) <= 1:
            return np.full(len(score), 0.5, dtype=float)
        ranks = pd.Series(score).rank(method="average").to_numpy(float) - 1.0
        return ranks / float(len(score) - 1)
    minimum = float(np.min(score))
    maximum = float(np.max(score))
    if maximum <= minimum:
        return np.full(len(score), 0.5, dtype=float)
    return (score - minimum) / (maximum - minimum)


def _grouping_summary(
    rows: pd.DataFrame,
    *,
    resolved_group_column: str | None,
    pair_config: CandidatePairForwardBackwardConfig,
    effective_config: CandidatePairForwardBackwardConfig,
    group_config: PairGroupMultiplicityConfig,
) -> dict[str, Any]:
    group_size = _numeric_column(rows, "candidate_pair_hypothesis_group_size")
    frame_count = (
        int(rows[["sequence_id", "time_s"]].drop_duplicates().shape[0])
        if not rows.empty
        else 0
    )
    group_count = (
        int(
            rows[
                ["sequence_id", "time_s", "candidate_pair_hypothesis_group"]
            ].drop_duplicates().shape[0]
        )
        if "candidate_pair_hypothesis_group" in rows.columns
        else 0
    )
    return {
        "resolved_group_column": resolved_group_column,
        "input_candidate_rows": int(len(rows)),
        "frame_count": frame_count,
        "physical_group_count": group_count,
        "duplicate_candidate_rows": int((group_size > 1).sum())
        if len(group_size)
        else 0,
        "group_size_mean": _safe_mean(group_size),
        "group_size_p95": _safe_quantile(group_size, 0.95),
        "group_size_max": _safe_max(group_size),
        "correction_strength": float(group_config.correction_strength),
        "original_score_column": pair_config.score_column,
        "original_score_normalization": pair_config.score_normalization,
        "effective_score_column": effective_config.score_column,
        "effective_score_normalization": effective_config.score_normalization,
    }


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _safe_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.quantile(float(quantile))) if len(finite) else float("nan")


def _safe_max(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite.loc[np.isfinite(finite)]
    return float(finite.max()) if len(finite) else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
