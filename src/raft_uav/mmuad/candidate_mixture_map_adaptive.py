"""Inference-safe adaptive top-K selection for MMUAD candidate-mixture MAP.

Fixed global top-K values trade recall against computation at every frame. A
small K is sufficient when one candidate has a clear score/uncertainty lead,
but ambiguous frames benefit from retaining more raw, dynamic, calibrated, and
merged hypotheses. This wrapper derives a per-frame K from score entropy,
score margin, learned sigma, and optional cross-sensor consensus, then applies
the existing branch/source-stratified selector before robust mixture-MAP.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
    CandidateMixtureMapResult,
    run_candidate_mixture_map,
    write_candidate_mixture_map_outputs,
)
from raft_uav.mmuad.candidate_mixture_map_stratified import (
    StratifiedMixtureTopKConfig,
    select_stratified_mixture_candidates,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns


@dataclass(frozen=True)
class AdaptiveMixtureTopKConfig:
    """Configuration for per-frame ambiguity-driven mixture top-K selection."""

    min_top_k: int = 3
    max_top_k: int = 20
    min_per_branch: int = 1
    min_per_source: int = 1
    min_per_source_branch: int = 0
    score_column: str = "candidate_reservoir_grid_score"
    fallback_score_columns: tuple[str, ...] = (
        "branch_consensus_rank_score",
        "ranker_score",
        "confidence",
    )
    sigma_column: str = "predicted_sigma_m"
    consensus_column: str = "branch_consensus_score"
    score_temperature: float = 0.25
    ambiguity_top_n: int = 20
    margin_scale: float = 0.25
    sigma_low_m: float = 2.0
    sigma_high_m: float = 30.0
    consensus_low: float = 0.0
    consensus_high: float = 1.0
    entropy_weight: float = 1.0
    margin_weight: float = 1.0
    sigma_weight: float = 1.0
    consensus_weight: float = 0.5
    branch_column: str = "candidate_branch"


@dataclass(frozen=True)
class AdaptiveCandidateMixtureMapResult:
    """Adaptive selected rows plus the downstream candidate-mixture result."""

    selected_candidates: pd.DataFrame
    mixture_result: CandidateMixtureMapResult
    selection_summary: dict[str, Any]


def select_adaptive_mixture_candidates(
    candidates: pd.DataFrame,
    *,
    config: AdaptiveMixtureTopKConfig | None = None,
) -> pd.DataFrame:
    """Select a branch/source-stratified, ambiguity-adaptive K per frame."""

    config = config or AdaptiveMixtureTopKConfig()
    _validate_config(config)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            mixture_adaptive_top_k=pd.Series(dtype=float),
            mixture_adaptive_ambiguity=pd.Series(dtype=float),
            mixture_adaptive_score_entropy=pd.Series(dtype=float),
            mixture_adaptive_margin_ambiguity=pd.Series(dtype=float),
            mixture_adaptive_sigma_ambiguity=pd.Series(dtype=float),
            mixture_adaptive_consensus_ambiguity=pd.Series(dtype=float),
        )
    rows = rows.copy().reset_index(drop=True)
    if "source" not in rows.columns:
        rows["source"] = "candidate"
    if config.branch_column not in rows.columns:
        rows[config.branch_column] = rows.get("candidate_branch", rows["source"])
    rows["source"] = rows["source"].fillna("candidate").astype(str)
    rows[config.branch_column] = (
        rows[config.branch_column].fillna("candidate").astype(str)
    )

    parts: list[pd.DataFrame] = []
    for _, frame in rows.groupby(
        ["sequence_id", "time_s"],
        sort=False,
        dropna=False,
    ):
        diagnostics = _frame_ambiguity(frame, config=config)
        adaptive_k = _adaptive_top_k(
            frame,
            diagnostics=diagnostics,
            config=config,
        )
        stratified = select_stratified_mixture_candidates(
            frame,
            config=StratifiedMixtureTopKConfig(
                top_k=adaptive_k,
                min_per_branch=int(config.min_per_branch),
                min_per_source=int(config.min_per_source),
                min_per_source_branch=int(config.min_per_source_branch),
                score_column=str(config.score_column),
                fallback_score_columns=tuple(config.fallback_score_columns),
                branch_column=str(config.branch_column),
                sigma_column=str(config.sigma_column),
            ),
        ).copy()
        stratified["mixture_adaptive_top_k"] = int(adaptive_k)
        stratified["mixture_adaptive_input_count"] = int(len(frame))
        stratified["mixture_adaptive_ambiguity_input_count"] = int(
            diagnostics["ambiguity_input_count"]
        )
        stratified["mixture_adaptive_ambiguity"] = diagnostics["ambiguity"]
        stratified["mixture_adaptive_score_entropy"] = diagnostics[
            "score_entropy"
        ]
        stratified["mixture_adaptive_margin_ambiguity"] = diagnostics[
            "margin_ambiguity"
        ]
        stratified["mixture_adaptive_sigma_ambiguity"] = diagnostics[
            "sigma_ambiguity"
        ]
        stratified["mixture_adaptive_consensus_ambiguity"] = diagnostics[
            "consensus_ambiguity"
        ]
        stratified["mixture_adaptive_available_signal_weight"] = diagnostics[
            "available_signal_weight"
        ]
        parts.append(stratified)
    out = pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()
    return out.sort_values(
        ["sequence_id", "time_s", "mixture_stratified_rank"],
        kind="mergesort",
    ).reset_index(drop=True)


def run_adaptive_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    adaptive_config: AdaptiveMixtureTopKConfig | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> AdaptiveCandidateMixtureMapResult:
    """Select adaptive top-K candidates and run robust candidate-mixture MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    adaptive_config = adaptive_config or AdaptiveMixtureTopKConfig(
        max_top_k=int(mixture_config.top_k),
        score_column=str(mixture_config.score_column),
        fallback_score_columns=tuple(mixture_config.fallback_score_columns),
        sigma_column=str(mixture_config.sigma_column),
    )
    selected = select_adaptive_mixture_candidates(
        candidates,
        config=adaptive_config,
    )
    effective_mixture_config = replace(
        mixture_config,
        top_k=int(adaptive_config.max_top_k),
    )
    mixture_result = run_candidate_mixture_map(
        selected,
        config=effective_mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    summary = build_adaptive_selection_summary(
        candidates,
        selected,
        adaptive_config=adaptive_config,
        mixture_config=effective_mixture_config,
    )
    return AdaptiveCandidateMixtureMapResult(
        selected_candidates=selected,
        mixture_result=mixture_result,
        selection_summary=summary,
    )


def build_adaptive_selection_summary(
    input_candidates: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    *,
    adaptive_config: AdaptiveMixtureTopKConfig,
    mixture_config: CandidateMixtureMapConfig,
) -> dict[str, Any]:
    """Build compact per-frame K and ambiguity diagnostics."""

    input_rows = normalize_candidate_columns(pd.DataFrame(input_candidates).copy())
    selected = normalize_candidate_columns(
        pd.DataFrame(selected_candidates).copy()
    )
    frame_rows = (
        selected.groupby(["sequence_id", "time_s"], sort=False)
        .first()
        .reset_index()
        if not selected.empty
        else pd.DataFrame()
    )
    top_k = pd.to_numeric(
        frame_rows.get("mixture_adaptive_top_k", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    ambiguity = pd.to_numeric(
        frame_rows.get("mixture_adaptive_ambiguity", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    selected_counts = (
        selected.groupby(["sequence_id", "time_s"], sort=False).size()
        if not selected.empty
        else pd.Series(dtype=float)
    )
    return {
        "input_candidate_rows": int(len(input_rows)),
        "selected_candidate_rows": int(len(selected)),
        "frame_count": int(len(frame_rows)),
        "adaptive_top_k_mean": _safe_mean(top_k),
        "adaptive_top_k_p50": _safe_quantile(top_k, 0.50),
        "adaptive_top_k_p95": _safe_quantile(top_k, 0.95),
        "adaptive_top_k_min": _safe_min(top_k),
        "adaptive_top_k_max": _safe_max(top_k),
        "ambiguity_mean": _safe_mean(ambiguity),
        "ambiguity_p95": _safe_quantile(ambiguity, 0.95),
        "selected_candidates_per_frame_mean": _safe_mean(selected_counts),
        "selected_candidates_per_frame_max": _safe_max(selected_counts),
        "adaptive_top_k_histogram": {
            str(int(key)): int(value)
            for key, value in top_k.astype(int).value_counts().sort_index().items()
        },
        "adaptive_config": asdict(adaptive_config),
        "mixture_config": asdict(mixture_config),
    }


def write_adaptive_candidate_mixture_outputs(
    result: AdaptiveCandidateMixtureMapResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write adaptive selection diagnostics and mixture-MAP outputs."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "mmuad_adaptive_mixture_candidates.csv"
    summary_path = output / "mmuad_adaptive_mixture_selection_summary.json"
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2),
        encoding="utf-8",
    )
    paths = write_candidate_mixture_map_outputs(result.mixture_result, output)
    paths["adaptive_candidates_csv"] = selected_path
    paths["adaptive_selection_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-adaptive-candidate-mixture-map",
        description=(
            "run ambiguity-adaptive stratified MMUAD candidate-mixture smoothing"
        ),
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--min-top-k", type=int, default=3)
    parser.add_argument("--max-top-k", type=int, default=20)
    parser.add_argument("--min-per-branch", type=int, default=1)
    parser.add_argument("--min-per-source", type=int, default=1)
    parser.add_argument("--min-per-source-branch", type=int, default=0)
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--consensus-column", default="branch_consensus_score")
    parser.add_argument("--adaptive-score-temperature", type=float, default=0.25)
    parser.add_argument("--adaptive-ambiguity-top-n", type=int, default=20)
    parser.add_argument("--adaptive-margin-scale", type=float, default=0.25)
    parser.add_argument("--adaptive-sigma-low-m", type=float, default=2.0)
    parser.add_argument("--adaptive-sigma-high-m", type=float, default=30.0)
    parser.add_argument("--adaptive-consensus-low", type=float, default=0.0)
    parser.add_argument("--adaptive-consensus-high", type=float, default=1.0)
    parser.add_argument("--adaptive-entropy-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-margin-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-sigma-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-consensus-weight", type=float, default=0.5)
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization",
        choices=SCORE_NORMALIZATION_CHOICES,
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization",
        choices=INITIALIZATION_CHOICES,
        default="uncertainty-top1",
    )
    args = parser.parse_args(argv)

    fallback_columns = tuple(args.fallback_score_column) or (
        "branch_consensus_rank_score",
        "ranker_score",
        "confidence",
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    initial_estimates = (
        None
        if args.initial_estimates_csv is None
        else pd.read_csv(args.initial_estimates_csv)
    )
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
    adaptive_config = AdaptiveMixtureTopKConfig(
        min_top_k=args.min_top_k,
        max_top_k=args.max_top_k,
        min_per_branch=args.min_per_branch,
        min_per_source=args.min_per_source,
        min_per_source_branch=args.min_per_source_branch,
        score_column=args.score_column,
        fallback_score_columns=fallback_columns,
        sigma_column=args.sigma_column,
        consensus_column=args.consensus_column,
        score_temperature=args.adaptive_score_temperature,
        ambiguity_top_n=args.adaptive_ambiguity_top_n,
        margin_scale=args.adaptive_margin_scale,
        sigma_low_m=args.adaptive_sigma_low_m,
        sigma_high_m=args.adaptive_sigma_high_m,
        consensus_low=args.adaptive_consensus_low,
        consensus_high=args.adaptive_consensus_high,
        entropy_weight=args.adaptive_entropy_weight,
        margin_weight=args.adaptive_margin_weight,
        sigma_weight=args.adaptive_sigma_weight,
        consensus_weight=args.adaptive_consensus_weight,
        branch_column=args.branch_column,
    )
    mixture_config = CandidateMixtureMapConfig(
        top_k=args.max_top_k,
        score_column=args.score_column,
        fallback_score_columns=fallback_columns,
        sigma_column=args.sigma_column,
        default_sigma_m=args.default_sigma_m,
        sigma_min_m=args.sigma_min_m,
        sigma_max_m=args.sigma_max_m,
        score_normalization=args.score_normalization,
        score_weight=args.score_weight,
        temperature=args.temperature,
        sigma_log_weight=args.sigma_log_weight,
        loss=args.loss,
        huber_delta=args.huber_delta,
        smoothness_weight=args.smoothness_weight,
        iterations=args.iterations,
        tolerance_m=args.tolerance_m,
        uniform_weight_floor=args.uniform_weight_floor,
        initialization=args.initialization,
    )
    result = run_adaptive_candidate_mixture_map(
        candidates,
        adaptive_config=adaptive_config,
        mixture_config=mixture_config,
        initial_estimates=initial_estimates,
        truth=truth,
    )
    paths = write_adaptive_candidate_mixture_outputs(result, args.output_dir)
    print("mmuad_adaptive_candidate_mixture_map=ok")
    print(f"candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    print(f"estimate_rows={len(result.mixture_result.estimates)}")
    print(f"adaptive_candidates_csv={paths['adaptive_candidates_csv']}")
    print(f"summary_json={paths['summary_json']}")
    return 0


def _frame_ambiguity(
    frame: pd.DataFrame,
    *,
    config: AdaptiveMixtureTopKConfig,
) -> dict[str, float]:
    score_all = _candidate_score(frame, config=config)
    top_n = min(int(config.ambiguity_top_n), len(frame))
    top_indices = np.argsort(score_all, kind="mergesort")[::-1][:top_n]
    score = score_all[top_indices]
    evaluation_frame = frame.iloc[top_indices]
    normalized = _minmax(score)
    probabilities = _softmax(
        normalized,
        temperature=config.score_temperature,
    )
    entropy = _normalized_entropy(probabilities)
    ordered = np.sort(normalized)[::-1]
    margin = float(ordered[0] - ordered[1]) if len(ordered) > 1 else 1.0
    margin_ambiguity = 1.0 - float(
        np.clip(margin / config.margin_scale, 0.0, 1.0)
    )

    sigma = _numeric_column(
        evaluation_frame,
        config.sigma_column,
        default=np.nan,
    ).to_numpy(float)
    finite_sigma = np.isfinite(sigma)
    sigma_ambiguity = float("nan")
    if finite_sigma.any():
        weights = _renormalized_subset(probabilities, finite_sigma)
        weighted_sigma = float(np.sum(weights * sigma[finite_sigma]))
        sigma_ambiguity = _scale_to_unit(
            weighted_sigma,
            low=config.sigma_low_m,
            high=config.sigma_high_m,
        )

    consensus = _numeric_column(
        evaluation_frame,
        config.consensus_column,
        default=np.nan,
    ).to_numpy(float)
    finite_consensus = np.isfinite(consensus)
    consensus_ambiguity = float("nan")
    if finite_consensus.any():
        weights = _renormalized_subset(probabilities, finite_consensus)
        weighted_consensus = float(
            np.sum(weights * consensus[finite_consensus])
        )
        consensus_strength = _scale_to_unit(
            weighted_consensus,
            low=config.consensus_low,
            high=config.consensus_high,
        )
        consensus_ambiguity = 1.0 - consensus_strength

    weighted_terms = [
        (config.entropy_weight, entropy),
        (config.margin_weight, margin_ambiguity),
        (config.sigma_weight, sigma_ambiguity),
        (config.consensus_weight, consensus_ambiguity),
    ]
    available = [
        (float(weight), float(value))
        for weight, value in weighted_terms
        if weight > 0 and np.isfinite(value)
    ]
    denominator = float(sum(weight for weight, _ in available))
    ambiguity = (
        float(sum(weight * value for weight, value in available) / denominator)
        if denominator > 0
        else 0.0
    )
    return {
        "ambiguity": float(np.clip(ambiguity, 0.0, 1.0)),
        "score_entropy": float(entropy),
        "margin_ambiguity": float(margin_ambiguity),
        "sigma_ambiguity": float(sigma_ambiguity),
        "consensus_ambiguity": float(consensus_ambiguity),
        "available_signal_weight": denominator,
        "ambiguity_input_count": float(top_n),
    }


def _adaptive_top_k(
    frame: pd.DataFrame,
    *,
    diagnostics: dict[str, float],
    config: AdaptiveMixtureTopKConfig,
) -> int:
    upper = min(int(config.max_top_k), len(frame))
    lower = min(int(config.min_top_k), upper)
    span = max(upper - lower, 0)
    adaptive = int(np.ceil(lower + diagnostics["ambiguity"] * span))
    branch_count = (
        frame[config.branch_column].fillna("candidate").astype(str).nunique()
    )
    source_count = frame["source"].fillna("candidate").astype(str).nunique()
    source_branch_count = (
        frame[["source", config.branch_column]]
        .fillna("candidate")
        .drop_duplicates()
        .shape[0]
    )
    quota_floor = max(
        lower,
        int(config.min_per_branch) * int(branch_count),
        int(config.min_per_source) * int(source_count),
        int(config.min_per_source_branch) * int(source_branch_count),
    )
    return int(np.clip(max(adaptive, quota_floor), lower, upper))


def _candidate_score(
    frame: pd.DataFrame,
    *,
    config: AdaptiveMixtureTopKConfig,
) -> np.ndarray:
    values = _numeric_column(frame, config.score_column, default=np.nan)
    for column in config.fallback_score_columns:
        values = values.fillna(
            _numeric_column(frame, column, default=np.nan)
        )
    return values.fillna(0.0).to_numpy(float)


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    default: float,
) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    out = np.zeros_like(values)
    minimum = float(np.min(values[finite]))
    maximum = float(np.max(values[finite]))
    if maximum > minimum:
        out[finite] = (values[finite] - minimum) / (maximum - minimum)
    else:
        out[finite] = 1.0
    return out


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    temperature = max(float(temperature), 1.0e-6)
    logits = np.asarray(values, dtype=float) / temperature
    logits = logits - float(np.max(logits))
    exp_values = np.exp(logits)
    total = float(np.sum(exp_values))
    if total > 0:
        return exp_values / total
    return np.full(len(logits), 1.0 / len(logits))


def _renormalized_subset(
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    weights = probabilities[mask]
    total = float(weights.sum())
    if total > 0:
        return weights / total
    return np.full(len(weights), 1.0 / len(weights))


def _normalized_entropy(probabilities: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) <= 1:
        return 0.0
    positive = probabilities[probabilities > 0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return float(entropy / np.log(len(probabilities)))


def _scale_to_unit(value: float, *, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("adaptive scale high must be greater than low")
    return float(
        np.clip(
            (float(value) - float(low)) / (float(high) - float(low)),
            0.0,
            1.0,
        )
    )


def _validate_config(config: AdaptiveMixtureTopKConfig) -> None:
    if int(config.min_top_k) <= 0:
        raise ValueError("min_top_k must be positive")
    if int(config.max_top_k) < int(config.min_top_k):
        raise ValueError("max_top_k must be at least min_top_k")
    if float(config.score_temperature) <= 0:
        raise ValueError("score_temperature must be positive")
    if int(config.ambiguity_top_n) <= 0:
        raise ValueError("ambiguity_top_n must be positive")
    if float(config.margin_scale) <= 0:
        raise ValueError("margin_scale must be positive")
    if float(config.sigma_high_m) <= float(config.sigma_low_m):
        raise ValueError("sigma_high_m must exceed sigma_low_m")
    if float(config.consensus_high) <= float(config.consensus_low):
        raise ValueError("consensus_high must exceed consensus_low")
    adaptive_weights = (
        config.entropy_weight,
        config.margin_weight,
        config.sigma_weight,
        config.consensus_weight,
    )
    if sum(max(0.0, float(value)) for value in adaptive_weights) <= 0:
        raise ValueError("at least one adaptive ambiguity weight must be positive")


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    return float(values.quantile(quantile)) if not values.empty else 0.0


def _safe_min(values: pd.Series) -> float:
    return float(values.min()) if not values.empty else 0.0


def _safe_max(values: pd.Series) -> float:
    return float(values.max()) if not values.empty else 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
