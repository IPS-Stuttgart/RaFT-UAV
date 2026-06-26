"""Soft global path priors for MMUAD candidate assignment.

The MMUAD branch-preserving pipeline can keep a good candidate in the reservoir
while still assigning it a weak per-frame ranker score. Hard Viterbi selection
then commits too early, whereas candidate-mixture MAP currently treats each
frame's candidate prior independently. This module adds a truth-free
forward-backward prior over the complete candidate sequence. It preserves soft
alternatives, rewards geometrically plausible transitions, and can feed the
resulting posterior directly into the existing robust candidate-mixture MAP
smoother.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    run_candidate_mixture_map,
    write_candidate_mixture_map_outputs,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import CandidateFrame, normalize_candidate_columns


SCORE_NORMALIZATION_CHOICES = ("minmax", "rank", "none")
DEFAULT_OUTPUT_SCORE_COLUMN = "candidate_forward_backward_score"


@dataclass(frozen=True)
class CandidateForwardBackwardConfig:
    """Configuration for the soft global candidate-path prior."""

    score_column: str = "candidate_reservoir_grid_score"
    fallback_score_columns: tuple[str, ...] = (
        "branch_consensus_rank_score",
        "candidate_temporal_consensus_score",
        "ranker_score",
        "confidence",
    )
    sigma_column: str = "predicted_sigma_m"
    default_sigma_m: float = 10.0
    sigma_min_m: float = 1.0
    sigma_max_m: float = 30.0
    score_normalization: str = "minmax"
    score_weight: float = 1.0
    sigma_log_weight: float = 1.0
    transition_distance_std_m: float = 2.0
    transition_speed_std_mps: float = 15.0
    max_speed_mps: float = 80.0
    speed_gate_penalty: float = 25.0
    source_switch_penalty: float = 0.25
    branch_switch_penalty: float = 0.25
    track_continuation_bonus: float = 0.5
    time_gap_penalty: float = 0.0
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN


def attach_forward_backward_candidate_prior(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    config: CandidateForwardBackwardConfig | None = None,
) -> CandidateFrame:
    """Attach frame-normalized forward-backward posterior probabilities.

    The posterior is inference-safe: it uses candidate geometry, timestamps,
    source/branch identity, candidate scores, and learned uncertainty only. No
    truth data is read by this function.
    """

    cfg = config or CandidateForwardBackwardConfig()
    _validate_config(cfg)
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)

    out = rows.copy().reset_index(drop=True)
    out["_fb_row_id"] = np.arange(len(out), dtype=int)
    out["candidate_branch"] = _branch_values(out)
    out["candidate_forward_backward_raw_score"] = _candidate_score(out, cfg)
    out["candidate_forward_backward_sigma_m"] = _candidate_sigma(out, cfg)
    _initialize_output_columns(out, cfg.output_score_column)

    for _, sequence in out.groupby("sequence_id", sort=False):
        _annotate_sequence(out, sequence, cfg)

    out = out.drop(columns=["_fb_row_id"], errors="ignore")
    return CandidateFrame(normalize_candidate_columns(out))


def forward_backward_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
) -> dict[str, Any]:
    """Return compact diagnostics for forward-backward augmented candidates."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return {
            "row_count": 0,
            "sequence_count": 0,
            "frame_count": 0,
            "score_column": score_column,
        }
    score = _numeric_column(rows, score_column)
    entropy = _numeric_column(rows, "candidate_forward_backward_frame_entropy")
    effective = _numeric_column(
        rows,
        "candidate_forward_backward_effective_candidate_count",
    )
    max_by_frame = (
        rows.assign(_posterior=score)
        .groupby(["sequence_id", "time_s"], sort=False)["_posterior"]
        .max()
    )
    top_rows = rows.loc[
        rows.groupby(["sequence_id", "time_s"], sort=False)[score_column].idxmax()
    ]
    return {
        "row_count": int(len(rows)),
        "sequence_count": int(rows["sequence_id"].astype(str).nunique()),
        "frame_count": int(rows[["sequence_id", "time_s"]].drop_duplicates().shape[0]),
        "score_column": str(score_column),
        "posterior_sum_error_max": _posterior_sum_error(rows, score_column),
        "top_posterior_mean": _safe_mean(max_by_frame),
        "top_posterior_p50": _safe_quantile(max_by_frame, 0.5),
        "top_posterior_p95": _safe_quantile(max_by_frame, 0.95),
        "frame_entropy_mean": _safe_mean(entropy),
        "effective_candidate_count_mean": _safe_mean(effective),
        "top_candidate_source_counts": _value_counts(top_rows, "source"),
        "top_candidate_branch_counts": _value_counts(top_rows, "candidate_branch"),
    }


def write_forward_backward_outputs(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    config: CandidateForwardBackwardConfig | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> None:
    """Write augmented candidate rows and optional provenance/summary JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if summary_json is None:
        return
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or CandidateForwardBackwardConfig()
    payload = {
        "config": asdict(cfg),
        "output_csv": str(output_csv),
        "summary": forward_backward_summary(
            candidates,
            score_column=cfg.output_score_column,
        ),
    }
    if extra_summary:
        payload.update(extra_summary)
    summary_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-forward-backward-prior",
        description=(
            "attach a soft global forward-backward path prior to MMUAD candidates"
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
        choices=SCORE_NORMALIZATION_CHOICES,
        default="minmax",
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=1.0)
    parser.add_argument("--transition-distance-std-m", type=float, default=2.0)
    parser.add_argument("--transition-speed-std-mps", type=float, default=15.0)
    parser.add_argument("--max-speed-mps", type=float, default=80.0)
    parser.add_argument("--speed-gate-penalty", type=float, default=25.0)
    parser.add_argument("--source-switch-penalty", type=float, default=0.25)
    parser.add_argument("--branch-switch-penalty", type=float, default=0.25)
    parser.add_argument("--track-continuation-bonus", type=float, default=0.5)
    parser.add_argument("--time-gap-penalty", type=float, default=0.0)
    parser.add_argument("--output-score-column", default=DEFAULT_OUTPUT_SCORE_COLUMN)
    parser.add_argument("--replace-confidence", action="store_true")

    parser.add_argument(
        "--mixture-output-dir",
        type=Path,
        help="optionally run the robust candidate-mixture MAP smoother on the posterior",
    )
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
        "branch_consensus_rank_score",
        "candidate_temporal_consensus_score",
        "ranker_score",
        "confidence",
    )
    cfg = CandidateForwardBackwardConfig(
        score_column=args.score_column,
        fallback_score_columns=fallback_columns,
        sigma_column=args.sigma_column,
        default_sigma_m=args.default_sigma_m,
        sigma_min_m=args.sigma_min_m,
        sigma_max_m=args.sigma_max_m,
        score_normalization=args.score_normalization,
        score_weight=args.score_weight,
        sigma_log_weight=args.sigma_log_weight,
        transition_distance_std_m=args.transition_distance_std_m,
        transition_speed_std_mps=args.transition_speed_std_mps,
        max_speed_mps=args.max_speed_mps,
        speed_gate_penalty=args.speed_gate_penalty,
        source_switch_penalty=args.source_switch_penalty,
        branch_switch_penalty=args.branch_switch_penalty,
        track_continuation_bonus=args.track_continuation_bonus,
        time_gap_penalty=args.time_gap_penalty,
        output_score_column=args.output_score_column,
    )
    augmented = attach_forward_backward_candidate_prior(
        load_candidate_file(args.candidate_csv),
        config=cfg,
    )
    if args.replace_confidence and not augmented.rows.empty:
        rows = augmented.rows.copy()
        rows["raw_confidence"] = pd.to_numeric(rows.get("confidence"), errors="coerce")
        rows["confidence"] = pd.to_numeric(rows[cfg.output_score_column], errors="coerce")
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
                top_k=args.mixture_top_k,
                score_column=cfg.output_score_column,
                fallback_score_columns=(cfg.score_column, *cfg.fallback_score_columns),
                sigma_column=cfg.sigma_column,
                default_sigma_m=cfg.default_sigma_m,
                sigma_min_m=cfg.sigma_min_m,
                sigma_max_m=cfg.sigma_max_m,
                score_normalization="none",
                score_weight=args.mixture_score_weight,
                temperature=args.mixture_temperature,
                loss="huber",
                huber_delta=args.mixture_huber_delta,
                smoothness_weight=args.mixture_smoothness_weight,
                iterations=args.mixture_iterations,
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

    write_forward_backward_outputs(
        augmented,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        config=cfg,
        extra_summary=extra_summary,
    )
    print("mmuad_forward_backward_candidate_prior=ok")
    print(f"output_csv={args.output_csv}")
    print(f"rows={len(augmented.rows)}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    if args.mixture_output_dir is not None:
        print(f"mixture_output_dir={args.mixture_output_dir}")
    return 0


def _validate_config(config: CandidateForwardBackwardConfig) -> None:
    if config.score_normalization not in SCORE_NORMALIZATION_CHOICES:
        raise ValueError(
            f"unsupported score normalization {config.score_normalization!r}"
        )
    if config.default_sigma_m <= 0.0:
        raise ValueError("default_sigma_m must be positive")
    if config.sigma_min_m <= 0.0 or config.sigma_max_m < config.sigma_min_m:
        raise ValueError("invalid sigma bounds")
    if config.transition_distance_std_m <= 0.0:
        raise ValueError("transition_distance_std_m must be positive")
    if config.transition_speed_std_mps < 0.0:
        raise ValueError("transition_speed_std_mps must be non-negative")
    if config.max_speed_mps <= 0.0:
        raise ValueError("max_speed_mps must be positive")
    if config.speed_gate_penalty < 0.0:
        raise ValueError("speed_gate_penalty must be non-negative")
    for value, name in (
        (config.source_switch_penalty, "source_switch_penalty"),
        (config.branch_switch_penalty, "branch_switch_penalty"),
        (config.track_continuation_bonus, "track_continuation_bonus"),
        (config.time_gap_penalty, "time_gap_penalty"),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative")


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
    finite = np.isfinite(rows[["time_s", "x_m", "y_m", "z_m"]].to_numpy(float)).all(axis=1)
    return rows.loc[finite].copy().reset_index(drop=True)


def _branch_values(rows: pd.DataFrame) -> pd.Series:
    if "candidate_branch" in rows.columns:
        branch = rows["candidate_branch"].fillna("").astype(str).str.strip()
    else:
        branch = pd.Series("", index=rows.index, dtype=str)
    source = rows.get("source", pd.Series("candidate", index=rows.index))
    source = source.fillna("candidate").astype(str).str.strip()
    return branch.where(branch.str.len() > 0, source).replace("", "candidate")


def _candidate_score(
    rows: pd.DataFrame,
    config: CandidateForwardBackwardConfig,
) -> pd.Series:
    for column in (config.score_column, *config.fallback_score_columns):
        if column in rows.columns:
            values = pd.to_numeric(rows[column], errors="coerce")
            if np.isfinite(values.to_numpy(float)).any():
                return values.fillna(float(values[np.isfinite(values)].min()))
    return pd.Series(1.0, index=rows.index, dtype=float)


def _candidate_sigma(
    rows: pd.DataFrame,
    config: CandidateForwardBackwardConfig,
) -> pd.Series:
    if config.sigma_column in rows.columns:
        sigma = pd.to_numeric(rows[config.sigma_column], errors="coerce")
    else:
        sigma = pd.Series(config.default_sigma_m, index=rows.index, dtype=float)
    sigma = sigma.fillna(config.default_sigma_m)
    return sigma.clip(config.sigma_min_m, config.sigma_max_m).astype(float)


def _initialize_output_columns(rows: pd.DataFrame, score_column: str) -> None:
    for column in (
        "candidate_forward_log_probability",
        "candidate_backward_log_probability",
        "candidate_forward_backward_log_probability",
        score_column,
        "candidate_forward_backward_rank",
        "candidate_forward_backward_frame_entropy",
        "candidate_forward_backward_effective_candidate_count",
        "candidate_forward_backward_best_previous_distance_m",
        "candidate_forward_backward_best_previous_speed_mps",
        "candidate_forward_backward_best_next_distance_m",
        "candidate_forward_backward_best_next_speed_mps",
    ):
        rows[column] = np.nan
    for column in (
        "candidate_forward_backward_best_previous_source",
        "candidate_forward_backward_best_previous_branch",
        "candidate_forward_backward_best_previous_track_id",
        "candidate_forward_backward_best_next_source",
        "candidate_forward_backward_best_next_branch",
        "candidate_forward_backward_best_next_track_id",
    ):
        rows[column] = ""


def _annotate_sequence(
    out: pd.DataFrame,
    sequence: pd.DataFrame,
    config: CandidateForwardBackwardConfig,
) -> None:
    frames = _sequence_frames(sequence, config)
    if not frames:
        return
    emissions = [frame["emission"] for frame in frames]
    transitions = [
        _transition_log_likelihood(frames[index - 1], frames[index], config)
        for index in range(1, len(frames))
    ]

    forward: list[np.ndarray] = [np.asarray(emissions[0], dtype=float)]
    forward[0] = forward[0] - _logsumexp(forward[0])
    for index in range(1, len(frames)):
        values = forward[index - 1][:, None] + transitions[index - 1]
        current = emissions[index] + _logsumexp(values, axis=0)
        forward.append(current - _logsumexp(current))

    backward: list[np.ndarray] = [np.zeros_like(emission) for emission in emissions]
    for index in range(len(frames) - 2, -1, -1):
        values = (
            transitions[index]
            + emissions[index + 1][None, :]
            + backward[index + 1][None, :]
        )
        current = _logsumexp(values, axis=1)
        backward[index] = current - _logsumexp(current)

    for index, frame in enumerate(frames):
        log_posterior = forward[index] + backward[index]
        log_posterior = log_posterior - _logsumexp(log_posterior)
        posterior = np.exp(log_posterior)
        entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1.0e-300))))
        ranks = _descending_ranks(posterior)
        indices = frame["indices"]
        out.loc[indices, "candidate_forward_log_probability"] = forward[index]
        out.loc[indices, "candidate_backward_log_probability"] = backward[index]
        out.loc[indices, "candidate_forward_backward_log_probability"] = log_posterior
        out.loc[indices, config.output_score_column] = posterior
        out.loc[indices, "candidate_forward_backward_rank"] = ranks
        out.loc[indices, "candidate_forward_backward_frame_entropy"] = entropy
        out.loc[
            indices,
            "candidate_forward_backward_effective_candidate_count",
        ] = float(np.exp(entropy))

        if index > 0:
            _write_best_previous(out, frames[index - 1], frame, forward[index - 1], transitions[index - 1])
        if index + 1 < len(frames):
            _write_best_next(
                out,
                frame,
                frames[index + 1],
                emissions[index + 1],
                backward[index + 1],
                transitions[index],
            )


def _sequence_frames(
    sequence: pd.DataFrame,
    config: CandidateForwardBackwardConfig,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for time_s, frame in sequence.groupby("time_s", sort=True):
        group = frame.copy()
        raw_score = pd.to_numeric(
            group["candidate_forward_backward_raw_score"],
            errors="coerce",
        ).to_numpy(float)
        normalized_score = _normalize_scores(raw_score, config.score_normalization)
        sigma = pd.to_numeric(
            group["candidate_forward_backward_sigma_m"],
            errors="coerce",
        ).to_numpy(float)
        emission = (
            float(config.score_weight) * normalized_score
            - float(config.sigma_log_weight) * np.log(sigma)
        )
        frames.append(
            {
                "time_s": float(time_s),
                "indices": group.index.to_numpy(int),
                "positions": group[["x_m", "y_m", "z_m"]].to_numpy(float),
                "sources": group["source"].fillna("candidate").astype(str).to_numpy(object),
                "branches": group["candidate_branch"].fillna("candidate").astype(str).to_numpy(object),
                "track_ids": group["track_id"].to_numpy(object),
                "emission": emission,
            }
        )
    return frames


def _transition_log_likelihood(
    previous: dict[str, Any],
    current: dict[str, Any],
    config: CandidateForwardBackwardConfig,
) -> np.ndarray:
    dt = max(float(current["time_s"] - previous["time_s"]), 1.0e-6)
    previous_xyz = np.asarray(previous["positions"], dtype=float)
    current_xyz = np.asarray(current["positions"], dtype=float)
    distance = np.linalg.norm(
        previous_xyz[:, None, :] - current_xyz[None, :, :],
        axis=2,
    )
    scale = float(
        np.hypot(
            config.transition_distance_std_m,
            config.transition_speed_std_mps * dt,
        )
    )
    cost = 0.5 * np.square(distance / max(scale, 1.0e-9))
    speed = distance / dt
    excess = np.maximum(speed / float(config.max_speed_mps) - 1.0, 0.0)
    cost = cost + float(config.speed_gate_penalty) * np.square(excess)
    cost = cost + float(config.time_gap_penalty) * dt

    previous_source = np.asarray(previous["sources"], dtype=object)
    current_source = np.asarray(current["sources"], dtype=object)
    cost = cost + float(config.source_switch_penalty) * (
        previous_source[:, None] != current_source[None, :]
    )
    previous_branch = np.asarray(previous["branches"], dtype=object)
    current_branch = np.asarray(current["branches"], dtype=object)
    cost = cost + float(config.branch_switch_penalty) * (
        previous_branch[:, None] != current_branch[None, :]
    )

    previous_track = np.asarray(previous["track_ids"], dtype=object)
    current_track = np.asarray(current["track_ids"], dtype=object)
    valid_previous = np.array([_valid_track_id(value) for value in previous_track])
    valid_current = np.array([_valid_track_id(value) for value in current_track])
    same_track = (
        valid_previous[:, None]
        & valid_current[None, :]
        & (previous_track[:, None].astype(str) == current_track[None, :].astype(str))
    )
    cost = cost - float(config.track_continuation_bonus) * same_track
    return -cost


def _write_best_previous(
    out: pd.DataFrame,
    previous: dict[str, Any],
    current: dict[str, Any],
    previous_forward: np.ndarray,
    transition: np.ndarray,
) -> None:
    scores = previous_forward[:, None] + transition
    best = np.argmax(scores, axis=0)
    dt = max(float(current["time_s"] - previous["time_s"]), 1.0e-6)
    distance = np.linalg.norm(
        np.asarray(previous["positions"])[best] - np.asarray(current["positions"]),
        axis=1,
    )
    indices = current["indices"]
    out.loc[indices, "candidate_forward_backward_best_previous_distance_m"] = distance
    out.loc[indices, "candidate_forward_backward_best_previous_speed_mps"] = distance / dt
    out.loc[indices, "candidate_forward_backward_best_previous_source"] = np.asarray(
        previous["sources"],
        dtype=object,
    )[best]
    out.loc[indices, "candidate_forward_backward_best_previous_branch"] = np.asarray(
        previous["branches"],
        dtype=object,
    )[best]
    out.loc[indices, "candidate_forward_backward_best_previous_track_id"] = [
        _track_text(value) for value in np.asarray(previous["track_ids"], dtype=object)[best]
    ]


def _write_best_next(
    out: pd.DataFrame,
    current: dict[str, Any],
    next_frame: dict[str, Any],
    next_emission: np.ndarray,
    next_backward: np.ndarray,
    transition: np.ndarray,
) -> None:
    scores = transition + next_emission[None, :] + next_backward[None, :]
    best = np.argmax(scores, axis=1)
    dt = max(float(next_frame["time_s"] - current["time_s"]), 1.0e-6)
    distance = np.linalg.norm(
        np.asarray(next_frame["positions"])[best] - np.asarray(current["positions"]),
        axis=1,
    )
    indices = current["indices"]
    out.loc[indices, "candidate_forward_backward_best_next_distance_m"] = distance
    out.loc[indices, "candidate_forward_backward_best_next_speed_mps"] = distance / dt
    out.loc[indices, "candidate_forward_backward_best_next_source"] = np.asarray(
        next_frame["sources"],
        dtype=object,
    )[best]
    out.loc[indices, "candidate_forward_backward_best_next_branch"] = np.asarray(
        next_frame["branches"],
        dtype=object,
    )[best]
    out.loc[indices, "candidate_forward_backward_best_next_track_id"] = [
        _track_text(value) for value in np.asarray(next_frame["track_ids"], dtype=object)[best]
    ]


def _normalize_scores(values: np.ndarray, mode: str) -> np.ndarray:
    score = np.asarray(values, dtype=float)
    finite = np.isfinite(score)
    if not finite.any():
        return np.ones(len(score), dtype=float)
    fill = float(np.min(score[finite]))
    score = np.where(finite, score, fill)
    if mode == "none":
        return score
    if mode == "rank":
        order = np.argsort(np.argsort(score, kind="stable"), kind="stable")
        return (order.astype(float) + 1.0) / max(len(score), 1)
    minimum = float(np.min(score))
    maximum = float(np.max(score))
    if maximum - minimum <= 1.0e-12:
        return np.ones(len(score), dtype=float)
    return (score - minimum) / (maximum - minimum)


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1, dtype=float)
    return ranks


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    maximum = np.max(array, axis=axis, keepdims=True)
    shifted = np.exp(array - maximum)
    result = maximum + np.log(np.sum(shifted, axis=axis, keepdims=True))
    if axis is None:
        return np.asarray(result).reshape(())
    return np.squeeze(result, axis=axis)


def _valid_track_id(value: object) -> bool:
    if value is None or pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "<na>"}


def _track_text(value: object) -> str:
    return str(value) if _valid_track_id(value) else ""


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _posterior_sum_error(rows: pd.DataFrame, score_column: str) -> float | None:
    if score_column not in rows.columns or rows.empty:
        return None
    sums = rows.groupby(["sequence_id", "time_s"], sort=False)[score_column].sum()
    if sums.empty:
        return None
    return float(np.max(np.abs(sums.to_numpy(float) - 1.0)))


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns or rows.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in rows[column].fillna("").astype(str).value_counts().items()
    }


def _safe_mean(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(float))]
    return None if finite.empty else float(finite.mean())


def _safe_quantile(values: pd.Series, q: float) -> float | None:
    finite = pd.to_numeric(values, errors="coerce")
    finite = finite[np.isfinite(finite.to_numpy(float))]
    return None if finite.empty else float(finite.quantile(float(q)))


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
