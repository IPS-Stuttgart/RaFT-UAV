"""Run multi-start candidate-mixture MAP with pair-prior log posteriors.

The pair-state forward-backward model writes both a posterior probability and a
log posterior for every candidate. Candidate-mixture MAP combines its score
column additively inside a log weight. Feeding a probability directly into
that expression compresses a strong posterior preference: for example, 0.9
versus 0.1 contributes only 0.8 score units, whereas the corresponding log
posterior contributes ``log(9)`` units.

This module provides an inference-safe ablation that normalizes the pair prior
within every frame, uses its log posterior as the mixture score, and retains the
existing learned-sigma, Huber, smoothness, and branch-seeded multi-start logic.
Truth remains optional and is used only for diagnostic metrics.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.candidate_mixture_map_multistart import (
    CandidateMixtureMultiStartConfig,
    CandidateMixtureMultiStartResult,
    run_multistart_candidate_mixture_map,
    write_multistart_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.io import load_candidate_csv, load_truth_csv
from raft_uav.mmuad.schema import normalize_candidate_columns
from raft_uav.mmuad.submission import load_sequence_class_map

DEFAULT_PAIR_PROBABILITY_COLUMN = "candidate_pair_forward_backward_score"
DEFAULT_PAIR_LOG_PROBABILITY_COLUMN = (
    "candidate_pair_forward_backward_log_probability"
)
DEFAULT_OUTPUT_SCORE_COLUMN = "candidate_pair_mixture_log_score"
ADAPTED_CANDIDATES_CSV = "mmuad_pair_log_multistart_candidates.csv"
COMBINED_SUMMARY_JSON = "mmuad_pair_log_multistart_summary.json"


@dataclass(frozen=True)
class PairLogPosteriorConfig:
    """Configuration for converting pair posteriors into mixture log scores."""

    probability_column: str = DEFAULT_PAIR_PROBABILITY_COLUMN
    log_probability_column: str = DEFAULT_PAIR_LOG_PROBABILITY_COLUMN
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN
    probability_floor: float = 1.0e-12


@dataclass(frozen=True)
class PairLogMultiStartResult:
    """Adapted candidates, selected multi-start result, and provenance."""

    candidates: pd.DataFrame
    multistart: CandidateMixtureMultiStartResult
    summary: dict[str, Any]


def attach_pair_log_posterior_score(
    candidates: pd.DataFrame,
    *,
    config: PairLogPosteriorConfig | None = None,
) -> pd.DataFrame:
    """Attach a frame-normalized log-posterior score for mixture inference.

    Existing finite log probabilities are preferred. Missing values are
    reconstructed from the probability column. Every frame is renormalized in
    log space, which makes the adapter robust to rounded CSV probabilities and
    preserves opaque sequence identifiers.
    """

    cfg = config or PairLogPosteriorConfig()
    _validate_pair_config(cfg)
    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy())
    if rows.empty:
        return rows.assign(
            **{
                cfg.output_score_column: pd.Series(dtype=float),
                "candidate_pair_mixture_probability": pd.Series(dtype=float),
                "candidate_pair_mixture_score_source": pd.Series(dtype=str),
            }
        )

    rows = rows.copy().reset_index(drop=True)
    rows["sequence_id"] = rows["sequence_id"].astype(str)
    output = pd.Series(np.nan, index=rows.index, dtype=float)
    probability_out = pd.Series(np.nan, index=rows.index, dtype=float)
    source_out = pd.Series("", index=rows.index, dtype=str)

    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        frame_log, source = _frame_log_posterior(frame, cfg)
        output.loc[frame.index] = frame_log
        probability_out.loc[frame.index] = np.exp(frame_log)
        source_out.loc[frame.index] = source

    rows[cfg.output_score_column] = output.astype(float)
    rows["candidate_pair_mixture_probability"] = probability_out.astype(float)
    rows["candidate_pair_mixture_score_source"] = source_out.astype(str)
    return normalize_candidate_columns(rows)


def pair_log_posterior_summary(
    candidates: pd.DataFrame,
    *,
    config: PairLogPosteriorConfig | None = None,
) -> dict[str, Any]:
    """Return compact normalization and score-distribution diagnostics."""

    cfg = config or PairLogPosteriorConfig()
    rows = pd.DataFrame(candidates).copy()
    if rows.empty:
        return {
            "row_count": 0,
            "frame_count": 0,
            "output_score_column": cfg.output_score_column,
        }
    score = pd.to_numeric(rows[cfg.output_score_column], errors="coerce")
    probability = pd.to_numeric(
        rows["candidate_pair_mixture_probability"],
        errors="coerce",
    )
    frame_sums = (
        rows.assign(_pair_probability=probability)
        .groupby(["sequence_id", "time_s"], sort=False)["_pair_probability"]
        .sum()
    )
    finite_score = score[np.isfinite(score.to_numpy(float))]
    return {
        "row_count": int(len(rows)),
        "sequence_count": int(rows["sequence_id"].astype(str).nunique()),
        "frame_count": int(len(frame_sums)),
        "output_score_column": cfg.output_score_column,
        "posterior_sum_error_max": (
            float(np.max(np.abs(frame_sums.to_numpy(float) - 1.0)))
            if len(frame_sums)
            else 0.0
        ),
        "log_score_min": _safe_stat(finite_score, "min"),
        "log_score_p50": _safe_stat(finite_score, "median"),
        "log_score_max": _safe_stat(finite_score, "max"),
        "score_source_counts": {
            str(key): int(value)
            for key, value in rows["candidate_pair_mixture_score_source"]
            .value_counts(dropna=False)
            .items()
        },
    }


def run_pair_log_multistart(
    candidates: pd.DataFrame,
    *,
    pair_log_config: PairLogPosteriorConfig | None = None,
    mixture_config: core.CandidateMixtureMapConfig | None = None,
    multistart_config: CandidateMixtureMultiStartConfig | None = None,
    external_initial_estimates: pd.DataFrame | None = None,
    truth: pd.DataFrame | None = None,
) -> PairLogMultiStartResult:
    """Run multi-start robust mixture-MAP using the pair log posterior."""

    pair_log_config = pair_log_config or PairLogPosteriorConfig()
    adapted = attach_pair_log_posterior_score(candidates, config=pair_log_config)
    mixture_config = mixture_config or core.CandidateMixtureMapConfig(
        top_k=0,
        score_column=pair_log_config.output_score_column,
        fallback_score_columns=(
            pair_log_config.probability_column,
            "ranker_score",
            "confidence",
        ),
        score_normalization="none",
        sigma_log_weight=0.0,
        loss="huber",
    )
    mixture_config = core.CandidateMixtureMapConfig(
        **{
            **asdict(mixture_config),
            "score_column": pair_log_config.output_score_column,
            "score_normalization": "none",
        }
    )
    multistart_config = multistart_config or CandidateMixtureMultiStartConfig()
    multistart = run_multistart_candidate_mixture_map(
        adapted,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external_initial_estimates,
        truth=truth,
    )
    summary = {
        "schema": "raft-uav-mmuad-pair-log-multistart-v1",
        "pair_log_config": asdict(pair_log_config),
        "mixture_config": asdict(mixture_config),
        "multistart_config": asdict(multistart_config),
        "pair_log_posterior": pair_log_posterior_summary(
            adapted,
            config=pair_log_config,
        ),
        "selected_start": multistart.selected_start,
        "multistart": multistart.summary,
        "truth_used_for_selection": False,
        "mixture_score_space": "log-posterior",
    }
    return PairLogMultiStartResult(
        candidates=adapted,
        multistart=multistart,
        summary=_jsonable(summary),
    )


def write_pair_log_multistart_outputs(
    result: PairLogMultiStartResult,
    *,
    output_dir: Path,
    class_map: Mapping[str, str] | None = None,
    default_classification: int | str = 2,
    official_results_csv: Path | None = None,
    official_zip: Path | None = None,
) -> dict[str, Path]:
    """Write adapted candidates, selected trajectory artifacts, and provenance."""

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
    candidate_path = output / ADAPTED_CANDIDATES_CSV
    result.candidates.to_csv(candidate_path, index=False)
    paths["adapted_candidates_csv"] = candidate_path
    summary_path = output / COMBINED_SUMMARY_JSON
    summary_path.write_text(json.dumps(_jsonable(result.summary), indent=2), encoding="utf-8")
    paths["combined_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.candidate_pair_log_multistart",
        description=(
            "run branch-seeded MMUAD mixture-MAP with pair forward-backward "
            "log-posterior scores"
        ),
    )
    parser.add_argument("--pair-candidates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--external-initial-estimates-csv", type=Path)
    parser.add_argument("--probability-column", default=DEFAULT_PAIR_PROBABILITY_COLUMN)
    parser.add_argument(
        "--log-probability-column",
        default=DEFAULT_PAIR_LOG_PROBABILITY_COLUMN,
    )
    parser.add_argument("--output-score-column", default=DEFAULT_OUTPUT_SCORE_COLUMN)
    parser.add_argument("--probability-floor", type=float, default=1.0e-12)

    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)

    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--max-branch-starts", type=int, default=8)
    parser.add_argument("--min-branch-frame-fraction", type=float, default=0.05)
    parser.add_argument("--no-score-top1-start", action="store_true")
    parser.add_argument("--no-frame-median-start", action="store_true")
    parser.add_argument("--no-branch-starts", action="store_true")

    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--default-classification", default="2")
    parser.add_argument("--official-results-csv", type=Path)
    parser.add_argument("--official-zip", type=Path)
    args = parser.parse_args(argv)

    pair_log_config = PairLogPosteriorConfig(
        probability_column=str(args.probability_column),
        log_probability_column=str(args.log_probability_column),
        output_score_column=str(args.output_score_column),
        probability_floor=float(args.probability_floor),
    )
    mixture_config = core.CandidateMixtureMapConfig(
        top_k=int(args.top_k),
        score_column=pair_log_config.output_score_column,
        fallback_score_columns=(
            pair_log_config.probability_column,
            "ranker_score",
            "confidence",
        ),
        sigma_column=str(args.sigma_column),
        default_sigma_m=float(args.default_sigma_m),
        sigma_min_m=float(args.sigma_min_m),
        sigma_max_m=float(args.sigma_max_m),
        score_normalization="none",
        score_weight=float(args.score_weight),
        temperature=float(args.temperature),
        sigma_log_weight=float(args.sigma_log_weight),
        loss="huber",
        huber_delta=float(args.huber_delta),
        smoothness_weight=float(args.smoothness_weight),
        anchor_weight=float(args.anchor_weight),
        iterations=int(args.iterations),
        uniform_weight_floor=float(args.uniform_weight_floor),
        branch_balance=float(args.branch_balance),
        source_balance=float(args.source_balance),
        responsibility_floor=float(args.responsibility_floor),
    )
    multistart_config = CandidateMixtureMultiStartConfig(
        branch_column=str(args.branch_column),
        include_score_top1=not bool(args.no_score_top1_start),
        include_frame_median=not bool(args.no_frame_median_start),
        include_branch_starts=not bool(args.no_branch_starts),
        max_branch_starts=int(args.max_branch_starts),
        min_branch_frame_fraction=float(args.min_branch_frame_fraction),
    )
    candidates = load_candidate_csv(args.pair_candidates_csv).rows
    truth = None if args.truth_csv is None else load_truth_csv(args.truth_csv).rows
    external = (
        None
        if args.external_initial_estimates_csv is None
        else read_estimate_csv(args.external_initial_estimates_csv)
    )
    result = run_pair_log_multistart(
        candidates,
        pair_log_config=pair_log_config,
        mixture_config=mixture_config,
        multistart_config=multistart_config,
        external_initial_estimates=external,
        truth=truth,
    )
    class_map = load_sequence_class_map(args.class_map) if args.class_map is not None else {}
    paths = write_pair_log_multistart_outputs(
        result,
        output_dir=args.output_dir,
        class_map=class_map,
        default_classification=args.default_classification,
        official_results_csv=args.official_results_csv,
        official_zip=args.official_zip,
    )
    print("mmuad_pair_log_multistart=ok")
    print(f"selected_start={result.multistart.selected_start}")
    print(f"candidate_rows={len(result.candidates)}")
    pooled = result.multistart.selected_result.summary.get("metrics", {}).get("pooled", {})
    if pooled.get("rmse_3d_m") is not None:
        print(f"diagnostic_rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _frame_log_posterior(
    frame: pd.DataFrame,
    config: PairLogPosteriorConfig,
) -> tuple[np.ndarray, str]:
    if frame.empty:
        return np.asarray([], dtype=float), "empty"
    log_values = pd.Series(np.nan, index=frame.index, dtype=float)
    if config.log_probability_column in frame.columns:
        log_values = pd.to_numeric(
            frame[config.log_probability_column],
            errors="coerce",
        )
    probability = pd.Series(np.nan, index=frame.index, dtype=float)
    if config.probability_column in frame.columns:
        probability = pd.to_numeric(frame[config.probability_column], errors="coerce")
    probability = probability.clip(lower=float(config.probability_floor), upper=1.0)
    reconstructed = np.log(probability)
    existing = np.isfinite(log_values.to_numpy(float))
    fallback = np.isfinite(reconstructed.to_numpy(float))
    combined = log_values.where(existing, reconstructed)
    if not np.isfinite(combined.to_numpy(float)).any():
        raise ValueError(
            "pair candidates need a finite log-probability or probability column: "
            f"{config.log_probability_column!r} / {config.probability_column!r}"
        )
    floor_log = float(np.log(config.probability_floor))
    values = combined.fillna(floor_log).to_numpy(float)
    normalized = values - _logsumexp(values)
    if bool(existing.all()):
        source = "log-probability"
    elif not bool(existing.any()) and bool(fallback.all()):
        source = "probability-fallback"
    else:
        source = "mixed"
    return normalized, source


def _validate_pair_config(config: PairLogPosteriorConfig) -> None:
    if not (0.0 < float(config.probability_floor) < 1.0):
        raise ValueError("probability_floor must be within (0, 1)")
    if not str(config.output_score_column).strip():
        raise ValueError("output_score_column must not be empty")


def _logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    return float(maximum + np.log(np.sum(np.exp(array - maximum))))


def _safe_stat(values: pd.Series, mode: str) -> float | None:
    if values.empty:
        return None
    if mode == "min":
        return float(values.min())
    if mode == "max":
        return float(values.max())
    return float(values.median())


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
