"""Build uncertainty-aware MMUAD candidate reservoirs.

The learned candidate uncertainty model is already used inside mixture-MAP, but
small top-K reservoirs are commonly formed from the ranker score alone.  That
can discard a slightly lower-scored candidate whose predicted geometric error
is much smaller.  This module combines candidate score and predicted sigma
before branch-preserving reservoir selection.

The default score has a probabilistic interpretation::

    risk_score = logit(p_good) - weight * log(sigma / sigma_floor)

where ``p_good`` is the calibrated/ranker probability and ``sigma`` is the
train-predicted candidate error scale.  No validation or test truth is needed
when applying the score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import (
    ReservoirConfig,
    build_candidate_reservoir,
    build_oracle_recall_tables,
    build_reservoir_summary,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns

RISK_SCORE_MODES = (
    "logit-minus-log-sigma",
    "probability-over-sigma",
    "score-minus-log-sigma",
)
DEFAULT_OUTPUT_SCORE_COLUMN = "candidate_risk_adjusted_score"
_EPS = 1.0e-6


def attach_candidate_risk_score(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    score_column: str = "candidate_class_calibrated_score",
    fallback_score_column: str = "ranker_score",
    sigma_column: str = "predicted_sigma_m",
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
    mode: str = "logit-minus-log-sigma",
    uncertainty_weight: float = 1.0,
    sigma_floor_m: float = 1.0,
) -> CandidateFrame:
    """Attach an uncertainty-aware candidate score without using truth labels."""

    mode = str(mode)
    if mode not in RISK_SCORE_MODES:
        raise ValueError(f"unsupported candidate risk score mode: {mode!r}")
    uncertainty_weight = _nonnegative_finite(
        uncertainty_weight,
        name="uncertainty_weight",
    )
    sigma_floor_m = _positive_finite(sigma_floor_m, name="sigma_floor_m")

    rows = _candidate_frame(candidates).rows.copy()
    if rows.empty:
        rows[output_score_column] = pd.Series(dtype=float)
        return CandidateFrame(normalize_candidate_columns(rows))

    base_score = _candidate_base_score(
        rows,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
    )
    probability = _score_probability(base_score)
    sigma, sigma_fallback = _candidate_sigma(
        rows,
        sigma_column=sigma_column,
        sigma_floor_m=sigma_floor_m,
    )
    relative_sigma = np.maximum(sigma / sigma_floor_m, 1.0)
    log_sigma_penalty = uncertainty_weight * np.log(relative_sigma)

    if mode == "logit-minus-log-sigma":
        risk_score = _logit(probability) - log_sigma_penalty
    elif mode == "probability-over-sigma":
        risk_score = probability / np.power(relative_sigma, uncertainty_weight)
    else:
        risk_score = base_score - log_sigma_penalty

    out = rows.copy()
    out["candidate_risk_base_score"] = base_score
    out["candidate_risk_probability"] = probability
    out["candidate_risk_sigma_m"] = sigma
    out["candidate_risk_log_sigma_penalty"] = log_sigma_penalty
    out["candidate_risk_sigma_fallback_m"] = float(sigma_fallback)
    out["candidate_risk_mode"] = mode
    out["candidate_risk_uncertainty_weight"] = uncertainty_weight
    out["candidate_risk_sigma_floor_m"] = sigma_floor_m
    out[output_score_column] = np.asarray(risk_score, dtype=float)
    return CandidateFrame(normalize_candidate_columns(out))


def build_risk_adjusted_reservoir(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    score_column: str = "candidate_class_calibrated_score",
    fallback_score_column: str = "ranker_score",
    sigma_column: str = "predicted_sigma_m",
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
    mode: str = "logit-minus-log-sigma",
    uncertainty_weight: float = 1.0,
    sigma_floor_m: float = 1.0,
    reservoir_config: ReservoirConfig | None = None,
) -> tuple[CandidateFrame, CandidateFrame]:
    """Attach risk scores and build a branch-preserving reservoir."""

    scored = attach_candidate_risk_score(
        candidates,
        score_column=score_column,
        fallback_score_column=fallback_score_column,
        sigma_column=sigma_column,
        output_score_column=output_score_column,
        mode=mode,
        uncertainty_weight=uncertainty_weight,
        sigma_floor_m=sigma_floor_m,
    )
    config = reservoir_config or ReservoirConfig(score_column=output_score_column)
    if config.score_column != output_score_column:
        config = ReservoirConfig(
            global_top_n=config.global_top_n,
            per_source_top_n=config.per_source_top_n,
            per_branch_top_n=config.per_branch_top_n,
            max_candidates_per_frame=config.max_candidates_per_frame,
            score_column=output_score_column,
            fallback_score_column=config.fallback_score_column,
            score_floor_quantile=config.score_floor_quantile,
            cap_reason_bonus=config.cap_reason_bonus,
        )
    reservoir_rows = build_candidate_reservoir(scored.rows, config=config)
    return scored, CandidateFrame(normalize_candidate_columns(reservoir_rows))


def risk_adjusted_reservoir_summary(
    scored: CandidateFrame | pd.DataFrame,
    reservoir: CandidateFrame | pd.DataFrame,
    *,
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
) -> dict[str, Any]:
    """Return compact provenance and distribution diagnostics."""

    scored_rows = _candidate_frame(scored).rows
    reservoir_rows = _candidate_frame(reservoir).rows
    summary = build_reservoir_summary(scored_rows, reservoir_rows)
    risk = pd.to_numeric(scored_rows.get(output_score_column), errors="coerce")
    sigma = pd.to_numeric(scored_rows.get("candidate_risk_sigma_m"), errors="coerce")
    probability = pd.to_numeric(scored_rows.get("candidate_risk_probability"), errors="coerce")
    summary.update(
        {
            "risk_score_column": str(output_score_column),
            "risk_score_mean": _finite_stat(risk, "mean"),
            "risk_score_p05": _finite_stat(risk, "p05"),
            "risk_score_p95": _finite_stat(risk, "p95"),
            "candidate_sigma_mean_m": _finite_stat(sigma, "mean"),
            "candidate_sigma_p95_m": _finite_stat(sigma, "p95"),
            "candidate_probability_mean": _finite_stat(probability, "mean"),
        }
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-risk-reservoir",
        description="build an uncertainty-aware branch-preserving MMUAD candidate reservoir",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--output-candidates-csv", type=Path, required=True)
    parser.add_argument("--output-reservoir-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--score-column", default="candidate_class_calibrated_score")
    parser.add_argument("--fallback-score-column", default="ranker_score")
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--output-score-column", default=DEFAULT_OUTPUT_SCORE_COLUMN)
    parser.add_argument("--mode", choices=RISK_SCORE_MODES, default="logit-minus-log-sigma")
    parser.add_argument("--uncertainty-weight", type=float, default=1.0)
    parser.add_argument("--sigma-floor-m", type=float, default=1.0)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--cap-reason-bonus", type=float, default=0.0)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--top-k", action="append", type=int, default=None)
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    candidates = load_candidate_file(args.candidates_csv)
    scored, reservoir = build_risk_adjusted_reservoir(
        candidates,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        sigma_column=args.sigma_column,
        output_score_column=args.output_score_column,
        mode=args.mode,
        uncertainty_weight=args.uncertainty_weight,
        sigma_floor_m=args.sigma_floor_m,
        reservoir_config=ReservoirConfig(
            global_top_n=args.global_top_n,
            per_source_top_n=args.per_source_top_n,
            per_branch_top_n=args.per_branch_top_n,
            max_candidates_per_frame=args.max_candidates_per_frame,
            score_column=args.output_score_column,
            fallback_score_column=args.fallback_score_column,
            score_floor_quantile=args.score_floor_quantile,
            cap_reason_bonus=args.cap_reason_bonus,
        ),
    )
    args.output_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_reservoir_csv.parent.mkdir(parents=True, exist_ok=True)
    scored.rows.to_csv(args.output_candidates_csv, index=False)
    reservoir.rows.to_csv(args.output_reservoir_csv, index=False)

    summary = risk_adjusted_reservoir_summary(
        scored,
        reservoir,
        output_score_column=args.output_score_column,
    )
    summary.update(
        {
            "mode": args.mode,
            "score_column": args.score_column,
            "fallback_score_column": args.fallback_score_column,
            "sigma_column": args.sigma_column,
            "uncertainty_weight": float(args.uncertainty_weight),
            "sigma_floor_m": float(args.sigma_floor_m),
        }
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.truth_csv is not None:
        truth = load_evaluation_truth_file(args.truth_csv).rows
        top_k_values = tuple(args.top_k) if args.top_k is not None else (1, 3, 5, 10, 20)
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            reservoir.rows,
            truth,
            top_k_values=top_k_values,
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        _write_optional_csv(frame_rows, args.oracle_frame_csv)
        _write_optional_csv(pooled, args.oracle_summary_csv)
        _write_optional_csv(by_sequence, args.oracle_by_sequence_csv)
        if not pooled.empty:
            print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")

    print("mmuad_candidate_risk_reservoir=ok")
    print(f"scored_candidates_csv={args.output_candidates_csv}")
    print(f"reservoir_csv={args.output_reservoir_csv}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    return 0


def _candidate_frame(candidates: CandidateFrame | pd.DataFrame) -> CandidateFrame:
    if isinstance(candidates, CandidateFrame):
        return CandidateFrame(normalize_candidate_columns(candidates.rows.copy()))
    return CandidateFrame(normalize_candidate_columns(pd.DataFrame(candidates).copy()))


def _nonnegative_finite(value: float, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return parsed


def _positive_finite(value: float, *, name: str) -> float:
    parsed = _finite_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _finite_float(value: float, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _candidate_base_score(
    rows: pd.DataFrame,
    *,
    score_column: str,
    fallback_score_column: str,
) -> np.ndarray:
    primary = _numeric_series(rows, score_column)
    fallback = _numeric_series(rows, fallback_score_column)
    primary = primary.where(np.isfinite(primary.to_numpy(float)), np.nan)
    fallback = fallback.where(np.isfinite(fallback.to_numpy(float)), np.nan)
    values = primary.where(primary.notna(), fallback).fillna(0.0)
    return values.to_numpy(float)


def _score_probability(score: np.ndarray) -> np.ndarray:
    finite = score[np.isfinite(score)]
    if finite.size and float(np.min(finite)) >= 0.0 and float(np.max(finite)) <= 1.0:
        return np.clip(score, _EPS, 1.0 - _EPS)
    clipped = np.clip(score, -30.0, 30.0)
    return np.clip(1.0 / (1.0 + np.exp(-clipped)), _EPS, 1.0 - _EPS)


def _candidate_sigma(
    rows: pd.DataFrame,
    *,
    sigma_column: str,
    sigma_floor_m: float,
) -> tuple[np.ndarray, float]:
    sigma_series = _numeric_series(rows, sigma_column)
    positive = sigma_series.loc[np.isfinite(sigma_series) & (sigma_series > 0.0)]
    fallback = float(positive.median()) if not positive.empty else float(sigma_floor_m)
    sigma = sigma_series.fillna(fallback).to_numpy(float)
    sigma = np.nan_to_num(sigma, nan=fallback, posinf=fallback, neginf=fallback)
    return np.maximum(sigma, float(sigma_floor_m)), fallback


def _numeric_series(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


def _finite_stat(values: pd.Series, statistic: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(float))]
    if finite.empty:
        return float("nan")
    if statistic == "mean":
        return float(finite.mean())
    if statistic == "p05":
        return float(finite.quantile(0.05))
    if statistic == "p95":
        return float(finite.quantile(0.95))
    raise ValueError(f"unsupported statistic: {statistic}")


def _write_optional_csv(rows: pd.DataFrame, path: Path | None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
