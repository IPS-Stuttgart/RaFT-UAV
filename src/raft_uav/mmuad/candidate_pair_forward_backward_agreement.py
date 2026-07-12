"""Agreement-adaptive pair-state priors for MMUAD candidate assignment.

The pair-state forward-backward posterior can be sharply concentrated on a smooth
but incorrect mode. Entropy alone then reports high confidence even when the
trajectory posterior strongly contradicts the local learned score/uncertainty
model. This module gates pair-posterior influence by both pair confidence and
local/global agreement measured with normalized Jensen-Shannon divergence.

Inference uses candidate scores, learned uncertainty, geometry, timestamps, and
metadata only. Optional truth is passed exclusively to downstream diagnostics.
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

DEFAULT_OUTPUT_SCORE_COLUMN = "candidate_pair_forward_backward_agreement_score"


@dataclass(frozen=True)
class AgreementAdaptivePairBlendConfig:
    """Configuration for confidence- and agreement-adaptive posterior blending."""

    min_pair_weight: float = 0.0
    max_pair_weight: float = 1.0
    confidence_power: float = 1.0
    agreement_power: float = 1.0
    epsilon: float = 1.0e-12
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN


def blend_candidate_posteriors(
    local_posterior: np.ndarray,
    pair_posterior: np.ndarray,
    *,
    config: AgreementAdaptivePairBlendConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Geometrically blend local and pair posteriors using confidence and agreement.

    Pair confidence is one minus normalized pair entropy. Agreement is one minus
    normalized Jensen-Shannon divergence. With ``agreement_power=0`` this reduces
    exactly to entropy-only adaptive blending.
    """

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
            "js_divergence": 0.0,
            "normalized_js_divergence": 0.0,
            "agreement": 1.0,
            "effective_pair_weight": float(cfg.min_pair_weight),
        }

    entropy = float(-np.sum(pair * np.log(np.maximum(pair, cfg.epsilon))))
    normalized_entropy = 0.0 if len(pair) <= 1 else entropy / float(np.log(len(pair)))
    normalized_entropy = float(np.clip(normalized_entropy, 0.0, 1.0))
    confidence = float(np.clip(1.0 - normalized_entropy, 0.0, 1.0))

    js_divergence = _jensen_shannon_divergence(local, pair, epsilon=cfg.epsilon)
    normalized_js = float(np.clip(js_divergence / np.log(2.0), 0.0, 1.0))
    agreement = float(np.clip(1.0 - normalized_js, 0.0, 1.0))
    trust = float(
        np.power(confidence, cfg.confidence_power)
        * np.power(agreement, cfg.agreement_power)
    )
    pair_weight = float(
        cfg.min_pair_weight
        + (cfg.max_pair_weight - cfg.min_pair_weight) * np.clip(trust, 0.0, 1.0)
    )
    log_blend = (
        (1.0 - pair_weight) * np.log(np.maximum(local, cfg.epsilon))
        + pair_weight * np.log(np.maximum(pair, cfg.epsilon))
    )
    blended = _stable_softmax(log_blend)
    return blended, {
        "pair_entropy": entropy,
        "pair_normalized_entropy": normalized_entropy,
        "pair_confidence": confidence,
        "js_divergence": js_divergence,
        "normalized_js_divergence": normalized_js,
        "agreement": agreement,
        "effective_pair_weight": pair_weight,
    }


def attach_agreement_adaptive_pair_prior(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    blend_config: AgreementAdaptivePairBlendConfig | None = None,
) -> CandidateFrame:
    """Attach an agreement-adaptive pair-state posterior to candidate rows."""

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    blend_cfg = blend_config or AgreementAdaptivePairBlendConfig()
    _validate_blend_config(blend_cfg)
    augmented = attach_pair_forward_backward_candidate_prior(candidates, config=pair_cfg)
    rows = augmented.rows.copy()
    if rows.empty:
        return CandidateFrame(normalize_candidate_columns(rows))

    _initialize_output_columns(rows, blend_cfg.output_score_column)
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        indices = frame.index
        local = _local_emission_posterior(frame, pair_cfg)
        pair = pd.to_numeric(frame[pair_cfg.output_score_column], errors="coerce").to_numpy(
            float
        )
        blended, diagnostics = blend_candidate_posteriors(
            local,
            pair,
            config=blend_cfg,
        )
        rows.loc[indices, "candidate_pair_forward_backward_local_posterior"] = local
        rows.loc[indices, blend_cfg.output_score_column] = blended
        rows.loc[indices, "candidate_pair_forward_backward_agreement_rank"] = (
            _descending_ranks(blended)
        )
        for suffix, key in (
            ("pair_entropy", "pair_entropy"),
            ("normalized_entropy", "pair_normalized_entropy"),
            ("pair_confidence", "pair_confidence"),
            ("js_divergence", "js_divergence"),
            ("normalized_js_divergence", "normalized_js_divergence"),
            ("agreement", "agreement"),
            ("pair_weight", "effective_pair_weight"),
        ):
            rows.loc[
                indices,
                f"candidate_pair_forward_backward_agreement_{suffix}",
            ] = diagnostics[key]

    return CandidateFrame(normalize_candidate_columns(rows))


def agreement_adaptive_pair_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
) -> dict[str, Any]:
    """Return compact framewise agreement-adaptive posterior diagnostics."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return {
            "row_count": 0,
            "sequence_count": 0,
            "frame_count": 0,
            "output_score_column": str(output_score_column),
        }
    first = rows.groupby(["sequence_id", "time_s"], sort=False).first().reset_index()
    weight = _numeric_column(first, "candidate_pair_forward_backward_agreement_pair_weight")
    confidence = _numeric_column(
        first,
        "candidate_pair_forward_backward_agreement_pair_confidence",
    )
    agreement = _numeric_column(first, "candidate_pair_forward_backward_agreement_agreement")
    js = _numeric_column(
        first,
        "candidate_pair_forward_backward_agreement_normalized_js_divergence",
    )
    adaptive_top = _top_candidate_ids(rows, output_score_column)
    local_top = _top_candidate_ids(rows, "candidate_pair_forward_backward_local_posterior")
    pair_top = _top_candidate_ids(rows, "candidate_pair_forward_backward_score")
    return {
        "row_count": int(len(rows)),
        "sequence_count": int(rows["sequence_id"].astype(str).nunique()),
        "frame_count": int(len(first)),
        "output_score_column": str(output_score_column),
        "posterior_sum_error_max": _posterior_sum_error(rows, output_score_column),
        "effective_pair_weight_mean": _safe_mean(weight),
        "effective_pair_weight_p50": _safe_quantile(weight, 0.50),
        "effective_pair_weight_p95": _safe_quantile(weight, 0.95),
        "pair_confidence_mean": _safe_mean(confidence),
        "agreement_mean": _safe_mean(agreement),
        "normalized_js_divergence_mean": _safe_mean(js),
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
    """Write augmented candidates and optional inference-provenance JSON."""

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
        "agreement_summary": agreement_adaptive_pair_summary(
            candidates,
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
        prog="mmuad-candidate-pair-forward-backward-agreement",
        description=(
            "blend local and pair-state MMUAD candidate posteriors using entropy "
            "confidence and Jensen-Shannon agreement"
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
    parser.add_argument("--confidence-power", type=float, default=1.0)
    parser.add_argument("--agreement-power", type=float, default=1.0)
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
        confidence_power=float(args.confidence_power),
        agreement_power=float(args.agreement_power),
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
        rows["confidence"] = pd.to_numeric(rows[blend_cfg.output_score_column], errors="coerce")
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


def _jensen_shannon_divergence(
    left: np.ndarray,
    right: np.ndarray,
    *,
    epsilon: float,
) -> float:
    mixture = 0.5 * (left + right)
    left_kl = np.sum(left * (np.log(np.maximum(left, epsilon)) - np.log(mixture)))
    right_kl = np.sum(right * (np.log(np.maximum(right, epsilon)) - np.log(mixture)))
    return float(max(0.0, 0.5 * (left_kl + right_kl)))


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


def _initialize_output_columns(rows: pd.DataFrame, output_score_column: str) -> None:
    for column in (
        "candidate_pair_forward_backward_local_posterior",
        output_score_column,
        "candidate_pair_forward_backward_agreement_rank",
        "candidate_pair_forward_backward_agreement_pair_entropy",
        "candidate_pair_forward_backward_agreement_normalized_entropy",
        "candidate_pair_forward_backward_agreement_pair_confidence",
        "candidate_pair_forward_backward_agreement_js_divergence",
        "candidate_pair_forward_backward_agreement_normalized_js_divergence",
        "candidate_pair_forward_backward_agreement_agreement",
        "candidate_pair_forward_backward_agreement_pair_weight",
    ):
        rows[column] = np.nan


def _validate_blend_config(config: AgreementAdaptivePairBlendConfig) -> None:
    for name in ("min_pair_weight", "max_pair_weight"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and within [0, 1]")
    if float(config.min_pair_weight) > float(config.max_pair_weight):
        raise ValueError("min_pair_weight must not exceed max_pair_weight")
    for name in ("confidence_power", "agreement_power"):
        value = float(getattr(config, name))
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not np.isfinite(float(config.epsilon)) or float(config.epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if not str(config.output_score_column).strip():
        raise ValueError("output_score_column must be non-empty")


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
    return _normalize_probability(np.exp(shifted))


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1, dtype=int)
    return ranks


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    rows = candidates.rows.copy() if isinstance(candidates, CandidateFrame) else pd.DataFrame(candidates)
    return normalize_candidate_columns(rows.copy())


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _posterior_sum_error(rows: pd.DataFrame, score_column: str) -> float:
    if score_column not in rows.columns:
        return float("nan")
    sums = rows.groupby(["sequence_id", "time_s"], sort=False)[score_column].sum()
    return float(np.max(np.abs(pd.to_numeric(sums, errors="coerce") - 1.0)))


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
        return float("nan")
    return float(np.mean(left.loc[common].to_numpy() != right.loc[common].to_numpy()))


def _safe_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else float("nan")


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.quantile(quantile)) if not finite.empty else float("nan")


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
