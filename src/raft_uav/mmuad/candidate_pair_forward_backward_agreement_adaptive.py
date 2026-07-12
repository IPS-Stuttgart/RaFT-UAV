"""Agreement-gated pair-state forward-backward priors for MMUAD candidates.

The acceleration-aware pair-state posterior can recover candidates buried by a
local ranker, but a smooth distractor can also produce a sharp global posterior.
Entropy alone cannot distinguish a correct decisive posterior from a confident
posterior that strongly contradicts the local learned score/uncertainty model.

This module therefore gates the pair-state influence by both posterior entropy
and local/global agreement. Agreement is measured as one minus the normalized
Jensen-Shannon divergence, yielding an inference-safe blend that backs off when
the temporal expert is confident but contradictory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


DEFAULT_OUTPUT_SCORE_COLUMN = "candidate_pair_forward_backward_agreement_adaptive_score"


@dataclass(frozen=True)
class AgreementAdaptivePairBlendConfig:
    """Configuration for entropy- and agreement-gated posterior blending."""

    min_pair_weight: float = 0.0
    max_pair_weight: float = 1.0
    entropy_power: float = 1.0
    agreement_power: float = 1.0
    agreement_floor: float = 0.0
    epsilon: float = 1.0e-12
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN


def blend_candidate_posteriors(
    local_posterior: np.ndarray,
    pair_posterior: np.ndarray,
    *,
    config: AgreementAdaptivePairBlendConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Blend local and pair posteriors using entropy and JS agreement gates."""

    cfg = config or AgreementAdaptivePairBlendConfig()
    _validate_blend_config(cfg)
    local = _normalize_probability(local_posterior)
    pair = _normalize_probability(pair_posterior)
    if len(local) != len(pair):
        raise ValueError("local and pair posteriors must have the same length")
    if len(local) == 0:
        return local, {
            "pair_entropy": 0.0,
            "pair_normalized_entropy": 0.0,
            "pair_confidence": 0.0,
            "local_pair_js_divergence": 0.0,
            "local_pair_agreement": 1.0,
            "agreement_factor": 1.0,
            "effective_pair_weight": float(cfg.min_pair_weight),
        }

    entropy = float(-np.sum(pair * np.log(np.maximum(pair, cfg.epsilon))))
    normalized_entropy = (
        0.0
        if len(pair) <= 1
        else float(np.clip(entropy / np.log(len(pair)), 0.0, 1.0))
    )
    pair_confidence = float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))
    js_divergence = _normalized_jensen_shannon_divergence(
        local,
        pair,
        epsilon=cfg.epsilon,
    )
    agreement = float(np.clip(1.0 - js_divergence, 0.0, 1.0))
    agreement_factor = float(
        cfg.agreement_floor
        + (1.0 - cfg.agreement_floor) * np.power(agreement, cfg.agreement_power)
    )
    pair_weight = float(
        cfg.min_pair_weight
        + (cfg.max_pair_weight - cfg.min_pair_weight)
        * np.power(pair_confidence, cfg.entropy_power)
        * agreement_factor
    )
    log_blend = (
        (1.0 - pair_weight) * np.log(np.maximum(local, cfg.epsilon))
        + pair_weight * np.log(np.maximum(pair, cfg.epsilon))
    )
    blended = _stable_softmax(log_blend)
    return blended, {
        "pair_entropy": entropy,
        "pair_normalized_entropy": normalized_entropy,
        "pair_confidence": pair_confidence,
        "local_pair_js_divergence": js_divergence,
        "local_pair_agreement": agreement,
        "agreement_factor": agreement_factor,
        "effective_pair_weight": pair_weight,
    }


def attach_agreement_adaptive_pair_prior(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    blend_config: AgreementAdaptivePairBlendConfig | None = None,
) -> CandidateFrame:
    """Attach an entropy- and agreement-adaptive pair-state posterior."""

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    blend_cfg = blend_config or AgreementAdaptivePairBlendConfig()
    _validate_blend_config(blend_cfg)
    augmented = attach_pair_forward_backward_candidate_prior(
        candidates,
        config=pair_cfg,
    )
    rows = augmented.rows.copy()
    if rows.empty:
        return CandidateFrame(normalize_candidate_columns(rows))

    _initialize_output_columns(rows, blend_cfg.output_score_column)
    pair_score_column = pair_cfg.output_score_column
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        indices = frame.index
        local = _local_emission_posterior(frame, pair_cfg)
        pair = pd.to_numeric(frame[pair_score_column], errors="coerce").to_numpy(float)
        blended, diagnostics = blend_candidate_posteriors(
            local,
            pair,
            config=blend_cfg,
        )
        rows.loc[indices, "candidate_pair_forward_backward_agreement_local_posterior"] = (
            local
        )
        rows.loc[indices, blend_cfg.output_score_column] = blended
        rows.loc[
            indices,
            "candidate_pair_forward_backward_agreement_adaptive_rank",
        ] = _descending_ranks(blended)
        for name, value in diagnostics.items():
            rows.loc[
                indices,
                f"candidate_pair_forward_backward_agreement_{name}",
            ] = value

    return CandidateFrame(normalize_candidate_columns(rows))


def agreement_adaptive_pair_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_score_column: str = "candidate_pair_forward_backward_score",
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
) -> dict[str, Any]:
    """Return compact diagnostics for agreement-adaptive posteriors."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return {
            "row_count": 0,
            "sequence_count": 0,
            "frame_count": 0,
            "pair_score_column": str(pair_score_column),
            "output_score_column": str(output_score_column),
        }
    frame_first = (
        rows.groupby(["sequence_id", "time_s"], sort=False).first().reset_index()
    )
    pair_weight = _numeric_column(
        frame_first,
        "candidate_pair_forward_backward_agreement_effective_pair_weight",
    )
    confidence = _numeric_column(
        frame_first,
        "candidate_pair_forward_backward_agreement_pair_confidence",
    )
    agreement = _numeric_column(
        frame_first,
        "candidate_pair_forward_backward_agreement_local_pair_agreement",
    )
    js_divergence = _numeric_column(
        frame_first,
        "candidate_pair_forward_backward_agreement_local_pair_js_divergence",
    )
    adaptive_top = _top_candidate_ids(rows, output_score_column)
    local_top = _top_candidate_ids(
        rows,
        "candidate_pair_forward_backward_agreement_local_posterior",
    )
    pair_top = _top_candidate_ids(rows, pair_score_column)
    return {
        "row_count": int(len(rows)),
        "sequence_count": int(rows["sequence_id"].astype(str).nunique()),
        "frame_count": int(len(frame_first)),
        "pair_score_column": str(pair_score_column),
        "output_score_column": str(output_score_column),
        "posterior_sum_error_max": _posterior_sum_error(rows, output_score_column),
        "effective_pair_weight_mean": _safe_mean(pair_weight),
        "effective_pair_weight_p50": _safe_quantile(pair_weight, 0.50),
        "effective_pair_weight_p95": _safe_quantile(pair_weight, 0.95),
        "pair_confidence_mean": _safe_mean(confidence),
        "local_pair_agreement_mean": _safe_mean(agreement),
        "local_pair_agreement_p05": _safe_quantile(agreement, 0.05),
        "local_pair_js_divergence_mean": _safe_mean(js_divergence),
        "local_pair_js_divergence_p95": _safe_quantile(js_divergence, 0.95),
        "adaptive_top_differs_from_local_fraction": _different_fraction(
            adaptive_top,
            local_top,
        ),
        "adaptive_top_differs_from_pair_fraction": _different_fraction(
            adaptive_top,
            pair_top,
        ),
    }


def write_agreement_adaptive_pair_outputs(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    blend_config: AgreementAdaptivePairBlendConfig | None = None,
    extra_summary: Mapping[str, Any] | None = None,
) -> None:
    """Write augmented candidates and optional provenance JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if summary_json is None:
        return
    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    blend_cfg = blend_config or AgreementAdaptivePairBlendConfig()
    payload: dict[str, Any] = {
        "schema": "raft-uav-mmuad-agreement-adaptive-pair-forward-backward-v1",
        "pair_config": asdict(pair_cfg),
        "blend_config": asdict(blend_cfg),
        "output_csv": str(output_csv),
        "pair_summary": pair_forward_backward_summary(
            candidates,
            score_column=pair_cfg.output_score_column,
        ),
        "agreement_adaptive_summary": agreement_adaptive_pair_summary(
            candidates,
            pair_score_column=pair_cfg.output_score_column,
            output_score_column=blend_cfg.output_score_column,
        ),
        "truth_used_for_candidate_prior": False,
    }
    if extra_summary:
        payload.update(dict(extra_summary))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmuad-candidate-pair-forward-backward-agreement-adaptive",
        description=(
            "blend local and pair-state MMUAD candidate posteriors using entropy "
            "and Jensen-Shannon agreement"
        ),
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
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
    parser.add_argument("--pair-score-column", default="candidate_pair_forward_backward_score")
    parser.add_argument("--min-pair-weight", type=float, default=0.0)
    parser.add_argument("--max-pair-weight", type=float, default=1.0)
    parser.add_argument("--entropy-power", type=float, default=1.0)
    parser.add_argument("--agreement-power", type=float, default=1.0)
    parser.add_argument("--agreement-floor", type=float, default=0.0)
    parser.add_argument("--output-score-column", default=DEFAULT_OUTPUT_SCORE_COLUMN)
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
    blend_cfg = AgreementAdaptivePairBlendConfig(
        min_pair_weight=float(args.min_pair_weight),
        max_pair_weight=float(args.max_pair_weight),
        entropy_power=float(args.entropy_power),
        agreement_power=float(args.agreement_power),
        agreement_floor=float(args.agreement_floor),
        output_score_column=str(args.output_score_column),
    )
    augmented = attach_agreement_adaptive_pair_prior(
        load_candidate_file(args.candidate_csv),
        pair_config=pair_cfg,
        blend_config=blend_cfg,
    )
    if args.replace_confidence and not augmented.rows.empty:
        rows = augmented.rows.copy()
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence"), errors="coerce")
        rows["confidence"] = pd.to_numeric(
            rows[blend_cfg.output_score_column],
            errors="coerce",
        )
        augmented = CandidateFrame(normalize_candidate_columns(rows))

    extra_summary: dict[str, Any] = {
        "candidate_csv": str(args.candidate_csv),
        "replace_confidence": bool(args.replace_confidence),
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
                score_column=blend_cfg.output_score_column,
                fallback_score_columns=(pair_cfg.output_score_column, pair_cfg.score_column),
                sigma_column=pair_cfg.sigma_column,
                default_sigma_m=pair_cfg.default_sigma_m,
                sigma_min_m=pair_cfg.sigma_min_m,
                sigma_max_m=pair_cfg.sigma_max_m,
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

    write_agreement_adaptive_pair_outputs(
        augmented,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        pair_config=pair_cfg,
        blend_config=blend_cfg,
        extra_summary=extra_summary,
    )
    print("mmuad_agreement_adaptive_pair_forward_backward=ok")
    print(f"output_csv={args.output_csv}")
    print(f"rows={len(augmented.rows)}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    if args.mixture_output_dir is not None:
        print(f"mixture_output_dir={args.mixture_output_dir}")
    return 0


def _normalized_jensen_shannon_divergence(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> float:
    midpoint = 0.5 * (left + right)
    left_kl = np.sum(
        left * (np.log(np.maximum(left, epsilon)) - np.log(np.maximum(midpoint, epsilon)))
    )
    right_kl = np.sum(
        right * (np.log(np.maximum(right, epsilon)) - np.log(np.maximum(midpoint, epsilon)))
    )
    divergence = 0.5 * float(left_kl + right_kl)
    return float(np.clip(divergence / np.log(2.0), 0.0, 1.0))


def _initialize_output_columns(rows: pd.DataFrame, output_score_column: str) -> None:
    columns = (
        "candidate_pair_forward_backward_agreement_local_posterior",
        output_score_column,
        "candidate_pair_forward_backward_agreement_adaptive_rank",
        "candidate_pair_forward_backward_agreement_pair_entropy",
        "candidate_pair_forward_backward_agreement_pair_normalized_entropy",
        "candidate_pair_forward_backward_agreement_pair_confidence",
        "candidate_pair_forward_backward_agreement_local_pair_js_divergence",
        "candidate_pair_forward_backward_agreement_local_pair_agreement",
        "candidate_pair_forward_backward_agreement_agreement_factor",
        "candidate_pair_forward_backward_agreement_effective_pair_weight",
    )
    for column in columns:
        rows[column] = np.nan


def _local_emission_posterior(
    frame: pd.DataFrame,
    config: CandidatePairForwardBackwardConfig,
) -> np.ndarray:
    raw_score = pd.to_numeric(
        frame["candidate_pair_forward_backward_raw_score"],
        errors="coerce",
    ).to_numpy(float)
    normalized_score = _normalize_scores(raw_score, config.score_normalization)
    sigma = pd.to_numeric(
        frame["candidate_pair_forward_backward_sigma_m"],
        errors="coerce",
    ).to_numpy(float)
    sigma = np.clip(sigma, config.sigma_min_m, config.sigma_max_m)
    logits = (
        float(config.score_weight) * normalized_score
        - float(config.sigma_log_weight) * np.log(sigma)
    )
    return _stable_softmax(logits)


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
        order = np.argsort(np.argsort(score, kind="stable"), kind="stable")
        return order.astype(float) / float(len(score) - 1)
    minimum = float(np.min(score))
    maximum = float(np.max(score))
    if maximum <= minimum:
        return np.full(len(score), 0.5, dtype=float)
    return (score - minimum) / (maximum - minimum)


def _normalize_probability(values: np.ndarray) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if probability.ndim != 1:
        raise ValueError("posterior arrays must be one-dimensional")
    probability = np.nan_to_num(probability, nan=0.0, posinf=0.0, neginf=0.0)
    probability = np.maximum(probability, 0.0)
    total = float(np.sum(probability))
    if total <= 0.0:
        return np.full(len(probability), 1.0 / max(len(probability), 1), dtype=float)
    return probability / total


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if len(values) == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(len(values), 1.0 / len(values), dtype=float)
    floor = float(np.min(values[finite])) - 100.0
    values = np.where(finite, values, floor)
    shifted = np.clip(values - float(np.max(values)), -700.0, 0.0)
    exponent = np.exp(shifted)
    return _normalize_probability(exponent)


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1, dtype=int)
    return ranks


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = (
        candidates.rows.copy()
        if isinstance(candidates, CandidateFrame)
        else pd.DataFrame(candidates)
    )
    return normalize_candidate_columns(rows.copy())


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _top_candidate_ids(rows: pd.DataFrame, score_column: str) -> pd.Series:
    if score_column not in rows.columns:
        return pd.Series(dtype=str)
    scored = rows.copy()
    scored["_agreement_score"] = pd.to_numeric(scored[score_column], errors="coerce")
    scored["_agreement_row_id"] = np.arange(len(scored), dtype=int)
    top = (
        scored.sort_values(
            ["sequence_id", "time_s", "_agreement_score", "_agreement_row_id"],
            ascending=[True, True, False, True],
        )
        .groupby(["sequence_id", "time_s"], sort=False)
        .head(1)
    )
    key = top["sequence_id"].astype(str) + "|" + top["time_s"].astype(str)
    value = top.get("track_id", top["_agreement_row_id"]).astype(str)
    return pd.Series(value.to_numpy(), index=key.to_numpy(), dtype=str)


def _different_fraction(left: pd.Series, right: pd.Series) -> float:
    common = left.index.intersection(right.index)
    if len(common) == 0:
        return 0.0
    return float(np.mean(left.loc[common].to_numpy() != right.loc[common].to_numpy()))


def _posterior_sum_error(rows: pd.DataFrame, score_column: str) -> float:
    if score_column not in rows.columns or rows.empty:
        return 0.0
    sums = (
        rows.assign(_score=pd.to_numeric(rows[score_column], errors="coerce").fillna(0.0))
        .groupby(["sequence_id", "time_s"], sort=False)["_score"]
        .sum()
    )
    return float(np.max(np.abs(sums.to_numpy(float) - 1.0))) if len(sums) else 0.0


def _safe_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else 0.0


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.quantile(quantile)) if not finite.empty else 0.0


def _validate_blend_config(config: AgreementAdaptivePairBlendConfig) -> None:
    values = (
        config.min_pair_weight,
        config.max_pair_weight,
        config.entropy_power,
        config.agreement_power,
        config.agreement_floor,
        config.epsilon,
    )
    if not all(np.isfinite(float(value)) for value in values):
        raise ValueError("agreement-adaptive blend parameters must be finite")
    if not 0.0 <= float(config.min_pair_weight) <= float(config.max_pair_weight) <= 1.0:
        raise ValueError("pair weights must satisfy 0 <= min_pair_weight <= max_pair_weight <= 1")
    if float(config.entropy_power) <= 0.0:
        raise ValueError("entropy_power must be positive")
    if float(config.agreement_power) <= 0.0:
        raise ValueError("agreement_power must be positive")
    if not 0.0 <= float(config.agreement_floor) <= 1.0:
        raise ValueError("agreement_floor must be in [0, 1]")
    if float(config.epsilon) <= 0.0:
        raise ValueError("epsilon must be positive")
    if not str(config.output_score_column).strip():
        raise ValueError("output_score_column must not be empty")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
