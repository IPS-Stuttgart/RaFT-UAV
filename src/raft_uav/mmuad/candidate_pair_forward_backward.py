"""Acceleration-aware pair-state forward-backward priors for MMUAD candidates.

The existing first-order candidate forward-backward model rewards plausible
point-to-point transitions, but a smooth distractor switch can still look good
when each individual transition is feasible.  This module lifts the hidden state
to consecutive candidate pairs.  The resulting second-order model adds a
constant-velocity acceleration factor while retaining soft, per-frame candidate
posteriors for the robust mixture-MAP smoother.

Inference uses only candidate geometry, timestamps, source/branch identity,
ranker scores, and learned uncertainty.  Truth is optional and is used only by
the downstream mixture-MAP diagnostic path.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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
DEFAULT_OUTPUT_SCORE_COLUMN = "candidate_pair_forward_backward_score"


@dataclass(frozen=True)
class CandidatePairForwardBackwardConfig:
    """Configuration for the acceleration-aware pair-state candidate prior."""

    score_column: str = "candidate_reservoir_grid_score"
    fallback_score_columns: tuple[str, ...] = (
        "candidate_risk_adjusted_score",
        "candidate_forward_backward_score",
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
    acceleration_std_mps2: float = 20.0
    max_acceleration_mps2: float = 80.0
    acceleration_gate_penalty: float = 25.0
    source_switch_penalty: float = 0.25
    branch_switch_penalty: float = 0.25
    track_continuation_bonus: float = 0.5
    time_gap_penalty: float = 0.0
    output_score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN


def attach_pair_forward_backward_candidate_prior(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    config: CandidatePairForwardBackwardConfig | None = None,
) -> CandidateFrame:
    """Attach acceleration-aware soft candidate posteriors.

    For one-frame sequences this reduces to the normalized emission model.  For
    two-frame sequences it is a first-order pair model.  From the third frame
    onward, pair states ``(candidate[t-1], candidate[t])`` are connected by an
    acceleration factor over candidate triples.
    """

    cfg = config or CandidatePairForwardBackwardConfig()
    _validate_config(cfg)
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)

    out = rows.copy().reset_index(drop=True)
    out["_pair_fb_row_id"] = np.arange(len(out), dtype=int)
    out["candidate_branch"] = _branch_values(out)
    out["candidate_pair_forward_backward_raw_score"] = _candidate_score(out, cfg)
    out["candidate_pair_forward_backward_sigma_m"] = _candidate_sigma(out, cfg)
    _initialize_output_columns(out, cfg.output_score_column)

    for _, sequence in out.groupby("sequence_id", sort=False):
        _annotate_sequence(out, sequence, cfg)

    out = out.drop(columns=["_pair_fb_row_id"], errors="ignore")
    return CandidateFrame(normalize_candidate_columns(out))


def pair_forward_backward_summary(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    score_column: str = DEFAULT_OUTPUT_SCORE_COLUMN,
) -> dict[str, Any]:
    """Return compact diagnostics for pair-state posterior candidates."""

    rows = _candidate_rows(candidates)
    if rows.empty:
        return {
            "row_count": 0,
            "sequence_count": 0,
            "frame_count": 0,
            "score_column": str(score_column),
        }
    score = _numeric_column(rows, score_column)
    entropy = _numeric_column(rows, "candidate_pair_forward_backward_frame_entropy")
    acceleration = _numeric_column(
        rows,
        "candidate_pair_forward_backward_min_acceleration_mps2",
    )
    top_by_frame = (
        rows.assign(_pair_score=score)
        .groupby(["sequence_id", "time_s"], sort=False)["_pair_score"]
        .max()
    )
    return {
        "row_count": int(len(rows)),
        "sequence_count": int(rows["sequence_id"].astype(str).nunique()),
        "frame_count": int(rows[["sequence_id", "time_s"]].drop_duplicates().shape[0]),
        "score_column": str(score_column),
        "posterior_sum_error_max": _posterior_sum_error(rows, score_column),
        "top_posterior_mean": _safe_mean(top_by_frame),
        "top_posterior_p50": _safe_quantile(top_by_frame, 0.50),
        "top_posterior_p95": _safe_quantile(top_by_frame, 0.95),
        "frame_entropy_mean": _safe_mean(entropy),
        "min_acceleration_mean_mps2": _safe_mean(acceleration),
        "min_acceleration_p95_mps2": _safe_quantile(acceleration, 0.95),
        "candidate_branch_counts": _value_counts(rows, "candidate_branch"),
        "source_counts": _value_counts(rows, "source"),
    }


def write_pair_forward_backward_outputs(
    candidates: CandidateFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    config: CandidatePairForwardBackwardConfig | None = None,
    extra_summary: Mapping[str, Any] | None = None,
) -> None:
    """Write augmented candidate rows and optional provenance JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    candidates.rows.to_csv(output_csv, index=False)
    if summary_json is None:
        return
    cfg = config or CandidatePairForwardBackwardConfig()
    payload: dict[str, Any] = {
        "schema": "raft-uav-mmuad-candidate-pair-forward-backward-v1",
        "config": asdict(cfg),
        "output_csv": str(output_csv),
        "summary": pair_forward_backward_summary(
            candidates,
            score_column=cfg.output_score_column,
        ),
        "truth_used_for_candidate_prior": False,
    }
    if extra_summary:
        payload.update(dict(extra_summary))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-pair-forward-backward-prior",
        description="attach an acceleration-aware pair-state prior to MMUAD candidates",
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
    parser.add_argument("--acceleration-std-mps2", type=float, default=20.0)
    parser.add_argument("--max-acceleration-mps2", type=float, default=80.0)
    parser.add_argument("--acceleration-gate-penalty", type=float, default=25.0)
    parser.add_argument("--source-switch-penalty", type=float, default=0.25)
    parser.add_argument("--branch-switch-penalty", type=float, default=0.25)
    parser.add_argument("--track-continuation-bonus", type=float, default=0.5)
    parser.add_argument("--time-gap-penalty", type=float, default=0.0)
    parser.add_argument("--output-score-column", default=DEFAULT_OUTPUT_SCORE_COLUMN)
    parser.add_argument("--replace-confidence", action="store_true")

    parser.add_argument(
        "--mixture-output-dir",
        type=Path,
        help="optionally run robust mixture-MAP using the pair-state posterior",
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
        "candidate_risk_adjusted_score",
        "candidate_forward_backward_score",
        "branch_consensus_rank_score",
        "candidate_temporal_consensus_score",
        "ranker_score",
        "confidence",
    )
    cfg = CandidatePairForwardBackwardConfig(
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
        output_score_column=str(args.output_score_column),
    )
    augmented = attach_pair_forward_backward_candidate_prior(
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
                top_k=int(args.mixture_top_k),
                score_column=cfg.output_score_column,
                fallback_score_columns=(cfg.score_column, *cfg.fallback_score_columns),
                sigma_column=cfg.sigma_column,
                default_sigma_m=cfg.default_sigma_m,
                sigma_min_m=cfg.sigma_min_m,
                sigma_max_m=cfg.sigma_max_m,
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

    write_pair_forward_backward_outputs(
        augmented,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        config=cfg,
        extra_summary=extra_summary,
    )
    print("mmuad_pair_forward_backward_candidate_prior=ok")
    print(f"output_csv={args.output_csv}")
    print(f"rows={len(augmented.rows)}")
    if args.summary_json is not None:
        print(f"summary_json={args.summary_json}")
    if args.mixture_output_dir is not None:
        print(f"mixture_output_dir={args.mixture_output_dir}")
    return 0


def _validate_config(config: CandidatePairForwardBackwardConfig) -> None:
    if config.score_normalization not in SCORE_NORMALIZATION_CHOICES:
        raise ValueError(f"unsupported score normalization {config.score_normalization!r}")
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
    if config.acceleration_std_mps2 <= 0.0:
        raise ValueError("acceleration_std_mps2 must be positive")
    if config.max_acceleration_mps2 <= 0.0:
        raise ValueError("max_acceleration_mps2 must be positive")
    if config.acceleration_gate_penalty < 0.0:
        raise ValueError("acceleration_gate_penalty must be non-negative")
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


def _candidate_sigma(
    rows: pd.DataFrame,
    config: CandidatePairForwardBackwardConfig,
) -> pd.Series:
    if config.sigma_column in rows.columns:
        sigma = pd.to_numeric(rows[config.sigma_column], errors="coerce")
    else:
        sigma = pd.Series(config.default_sigma_m, index=rows.index, dtype=float)
    sigma = sigma.fillna(config.default_sigma_m)
    sigma = sigma.where(sigma > 0.0, config.default_sigma_m)
    return sigma.clip(config.sigma_min_m, config.sigma_max_m).astype(float)


def _initialize_output_columns(rows: pd.DataFrame, score_column: str) -> None:
    for column in (
        "candidate_pair_forward_backward_log_probability",
        score_column,
        "candidate_pair_forward_backward_rank",
        "candidate_pair_forward_backward_frame_entropy",
        "candidate_pair_forward_backward_effective_candidate_count",
        "candidate_pair_forward_backward_pair_state_count",
        "candidate_pair_forward_backward_min_acceleration_mps2",
    ):
        rows[column] = np.nan


def _annotate_sequence(
    out: pd.DataFrame,
    sequence: pd.DataFrame,
    config: CandidatePairForwardBackwardConfig,
) -> None:
    frames = _sequence_frames(sequence, config)
    if not frames:
        return
    if len(frames) == 1:
        log_posterior = frames[0]["emission"] - _logsumexp(frames[0]["emission"])
        _write_frame_posterior(out, frames[0], log_posterior, pair_state_count=0)
        return

    transitions = [
        _transition_log_likelihood(frames[index - 1], frames[index], config)
        for index in range(1, len(frames))
    ]
    accelerations: list[np.ndarray | None] = [None] * len(frames)
    acceleration_norms: list[np.ndarray | None] = [None] * len(frames)
    for index in range(2, len(frames)):
        log_factor, norm = _acceleration_log_likelihood(
            frames[index - 2],
            frames[index - 1],
            frames[index],
            config,
        )
        accelerations[index] = log_factor
        acceleration_norms[index] = norm

    forward: list[np.ndarray | None] = [None] * len(frames)
    initial_pair = (
        frames[0]["emission"][:, None]
        + transitions[0]
        + frames[1]["emission"][None, :]
    )
    forward[1] = initial_pair - _logsumexp(initial_pair)
    for index in range(2, len(frames)):
        previous = np.asarray(forward[index - 1], dtype=float)
        acceleration = np.asarray(accelerations[index], dtype=float)
        values = (
            previous[:, :, None]
            + acceleration
            + transitions[index - 1][None, :, :]
            + frames[index]["emission"][None, None, :]
        )
        current = _logsumexp(values, axis=0)
        forward[index] = current - _logsumexp(current)

    backward: list[np.ndarray | None] = [None] * len(frames)
    backward[-1] = np.zeros_like(np.asarray(forward[-1], dtype=float))
    for index in range(len(frames) - 2, 0, -1):
        acceleration = np.asarray(accelerations[index + 1], dtype=float)
        next_backward = np.asarray(backward[index + 1], dtype=float)
        values = (
            acceleration
            + transitions[index][None, :, :]
            + frames[index + 1]["emission"][None, None, :]
            + next_backward[None, :, :]
        )
        current = _logsumexp(values, axis=2)
        backward[index] = current - _logsumexp(current)

    first_pair_log = np.asarray(forward[1], dtype=float) + np.asarray(backward[1], dtype=float)
    first_pair_log = first_pair_log - _logsumexp(first_pair_log)
    first_log_posterior = _logsumexp(first_pair_log, axis=1)
    first_log_posterior = first_log_posterior - _logsumexp(first_log_posterior)
    _write_frame_posterior(
        out,
        frames[0],
        first_log_posterior,
        pair_state_count=int(first_pair_log.size),
    )

    for index in range(1, len(frames)):
        pair_log = np.asarray(forward[index], dtype=float) + np.asarray(backward[index], dtype=float)
        pair_log = pair_log - _logsumexp(pair_log)
        log_posterior = _logsumexp(pair_log, axis=0)
        log_posterior = log_posterior - _logsumexp(log_posterior)
        _write_frame_posterior(
            out,
            frames[index],
            log_posterior,
            pair_state_count=int(pair_log.size),
        )

    for index in range(1, len(frames) - 1):
        norm = np.asarray(acceleration_norms[index + 1], dtype=float)
        minimum = np.min(norm, axis=(0, 2))
        out.loc[
            frames[index]["indices"],
            "candidate_pair_forward_backward_min_acceleration_mps2",
        ] = minimum


def _write_frame_posterior(
    out: pd.DataFrame,
    frame: dict[str, Any],
    log_posterior: np.ndarray,
    *,
    pair_state_count: int,
) -> None:
    posterior = np.exp(np.asarray(log_posterior, dtype=float))
    posterior = posterior / max(float(np.sum(posterior)), 1.0e-300)
    entropy = float(-np.sum(posterior * np.log(np.maximum(posterior, 1.0e-300))))
    indices = np.asarray(frame["indices"], dtype=int)
    out.loc[indices, "candidate_pair_forward_backward_log_probability"] = np.log(
        np.maximum(posterior, 1.0e-300)
    )
    out.loc[indices, frame["output_score_column"]] = posterior
    out.loc[indices, "candidate_pair_forward_backward_rank"] = _descending_ranks(posterior)
    out.loc[indices, "candidate_pair_forward_backward_frame_entropy"] = entropy
    out.loc[
        indices,
        "candidate_pair_forward_backward_effective_candidate_count",
    ] = float(np.exp(entropy))
    out.loc[indices, "candidate_pair_forward_backward_pair_state_count"] = int(
        pair_state_count
    )


def _sequence_frames(
    sequence: pd.DataFrame,
    config: CandidatePairForwardBackwardConfig,
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for time_s, frame in sequence.groupby("time_s", sort=True):
        group = frame.copy()
        raw_score = pd.to_numeric(
            group["candidate_pair_forward_backward_raw_score"],
            errors="coerce",
        ).to_numpy(float)
        normalized_score = _normalize_scores(raw_score, config.score_normalization)
        sigma = pd.to_numeric(
            group["candidate_pair_forward_backward_sigma_m"],
            errors="coerce",
        ).to_numpy(float)
        emission = (
            float(config.score_weight) * normalized_score
            - float(config.sigma_log_weight) * np.log(sigma)
        )
        source = group.get("source", pd.Series("candidate", index=group.index))
        track = group.get("track_id", pd.Series("", index=group.index))
        frames.append(
            {
                "time_s": float(time_s),
                "indices": group.index.to_numpy(int),
                "positions": group[["x_m", "y_m", "z_m"]].to_numpy(float),
                "sources": source.fillna("candidate").astype(str).to_numpy(object),
                "branches": group["candidate_branch"].fillna("candidate").astype(str).to_numpy(object),
                "track_ids": track.to_numpy(object),
                "emission": emission,
                "output_score_column": config.output_score_column,
            }
        )
    return frames


def _transition_log_likelihood(
    previous: dict[str, Any],
    current: dict[str, Any],
    config: CandidatePairForwardBackwardConfig,
) -> np.ndarray:
    dt = max(float(current["time_s"] - previous["time_s"]), 1.0e-6)
    previous_xyz = np.asarray(previous["positions"], dtype=float)
    current_xyz = np.asarray(current["positions"], dtype=float)
    distance = np.linalg.norm(previous_xyz[:, None, :] - current_xyz[None, :, :], axis=2)
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


def _acceleration_log_likelihood(
    previous: dict[str, Any],
    current: dict[str, Any],
    following: dict[str, Any],
    config: CandidatePairForwardBackwardConfig,
) -> tuple[np.ndarray, np.ndarray]:
    previous_dt = max(float(current["time_s"] - previous["time_s"]), 1.0e-6)
    following_dt = max(float(following["time_s"] - current["time_s"]), 1.0e-6)
    previous_xyz = np.asarray(previous["positions"], dtype=float)
    current_xyz = np.asarray(current["positions"], dtype=float)
    following_xyz = np.asarray(following["positions"], dtype=float)
    incoming_velocity = (
        current_xyz[None, :, :] - previous_xyz[:, None, :]
    ) / previous_dt
    outgoing_velocity = (
        following_xyz[None, :, :] - current_xyz[:, None, :]
    ) / following_dt
    acceleration = (
        2.0
        * (outgoing_velocity[None, :, :, :] - incoming_velocity[:, :, None, :])
        / (previous_dt + following_dt)
    )
    acceleration_norm = np.linalg.norm(acceleration, axis=3)
    cost = 0.5 * np.square(acceleration_norm / float(config.acceleration_std_mps2))
    excess = np.maximum(
        acceleration_norm / float(config.max_acceleration_mps2) - 1.0,
        0.0,
    )
    cost = cost + float(config.acceleration_gate_penalty) * np.square(excess)
    return -cost, acceleration_norm


def _normalize_scores(values: np.ndarray, mode: str) -> np.ndarray:
    score = np.asarray(values, dtype=float)
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
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


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("-inf")
    maximum = np.max(array, axis=axis, keepdims=True)
    finite_maximum = np.isfinite(maximum)
    shifted = np.where(finite_maximum, array - maximum, float("-inf"))
    total = np.sum(np.exp(np.clip(shifted, -700.0, 0.0)), axis=axis, keepdims=True)
    result = np.where(finite_maximum, maximum + np.log(np.maximum(total, 1.0e-300)), maximum)
    if axis is None:
        return float(np.asarray(result).reshape(-1)[0])
    return np.squeeze(result, axis=axis)


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(-array, kind="stable")
    ranks = np.empty(len(array), dtype=int)
    ranks[order] = np.arange(1, len(array) + 1, dtype=int)
    return ranks


def _valid_track_id(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "nan", "none", "null", "-1"}


def _numeric_column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows.columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce")


def _posterior_sum_error(rows: pd.DataFrame, score_column: str) -> float:
    if rows.empty or score_column not in rows.columns:
        return float("nan")
    sums = (
        rows.assign(_posterior=pd.to_numeric(rows[score_column], errors="coerce").fillna(0.0))
        .groupby(["sequence_id", "time_s"], sort=False)["_posterior"]
        .sum()
    )
    return float(np.max(np.abs(sums.to_numpy(float) - 1.0))) if len(sums) else float("nan")


def _safe_mean(values: Sequence[float] | pd.Series) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def _safe_quantile(values: Sequence[float] | pd.Series, quantile: float) -> float:
    array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    array = array[np.isfinite(array)]
    return float(np.quantile(array, float(quantile))) if array.size else float("nan")


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in rows[column].fillna("unknown").astype(str).value_counts().items()
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
