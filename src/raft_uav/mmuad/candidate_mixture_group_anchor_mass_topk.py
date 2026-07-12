"""Anchor-conditioned adaptive hypothesis-group selection for MMUAD mixture-MAP.

The posterior-mass group selector is intentionally state independent. That is
safe, but it can still discard a low-unary physical hypothesis that is highly
consistent with an inference-time trajectory initialization before the robust
mixture smoother sees it. This module adds a bounded, Huber-robust anchor cost
to the existing score/uncertainty unary used only for group selection.

The final grouped candidate-mixture MAP run still uses the original candidate
score, learned uncertainty, Huber loss, and trajectory objective. Ground truth
is optional and is never used for selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad import candidate_mixture_map as core
from raft_uav.mmuad.candidate_mixture_group_mass_topk import (
    PosteriorMassGroupTopKConfig,
    select_posterior_mass_hypothesis_group_topk,
)
from raft_uav.mmuad.candidate_mixture_group_topk import GROUP_SCORE_MODES
from raft_uav.mmuad.candidate_mixture_map import (
    INITIALIZATION_CHOICES,
    LOSS_CHOICES,
    SCORE_NORMALIZATION_CHOICES,
    CandidateMixtureMapConfig,
)
from raft_uav.mmuad.candidate_mixture_map_grouped import (
    GroupedCandidateMixtureMapResult,
    HypothesisGroupConfig,
    run_grouped_candidate_mixture_map,
    write_grouped_candidate_mixture_outputs,
)
from raft_uav.mmuad.estimate_csv import read_estimate_csv
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import normalize_candidate_columns

ANCHOR_UTILITY_COLUMN = "mixture_anchor_conditioned_selection_utility"
MISSING_ANCHOR_POLICIES = ("neutral", "error")
_SEQUENCE_ALIASES = (
    "sequence_id",
    "Sequence",
    "sequence",
    "seq",
    "scene",
    "scene_id",
    "clip",
    "clip_id",
)
_TIME_ALIASES = ("time_s", "Timestamp", "timestamp", "time")
_X_ALIASES = ("state_x_m", "x_m")
_Y_ALIASES = ("state_y_m", "y_m")
_Z_ALIASES = ("state_z_m", "z_m")


@dataclass(frozen=True)
class AnchorConditioningConfig:
    """Configuration for robust trajectory-conditioned group selection."""

    anchor_selection_weight: float = 1.0
    anchor_scale_m: float = 10.0
    anchor_huber_delta: float = 1.0
    anchor_cost_cap: float = 4.0
    anchor_time_tolerance_s: float = 0.5
    missing_anchor_policy: str = "neutral"


@dataclass(frozen=True)
class AnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Anchor-scored candidates, selected groups, and final grouped MAP result."""

    scored_candidates: pd.DataFrame
    selected_candidates: pd.DataFrame
    grouped_result: GroupedCandidateMixtureMapResult
    selection_summary: dict[str, Any]


def add_anchor_conditioned_selection_utility(
    candidates: pd.DataFrame,
    initial_estimates: pd.DataFrame,
    *,
    mixture_config: CandidateMixtureMapConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Attach a bounded anchor-conditioned unary to every candidate row.

    The base unary exactly mirrors the maintained score/uncertainty utility:

    ``score_weight * normalized_score / temperature - sigma_log_weight * log(sigma)``.

    A Huber cost of distance to the interpolated initialization is then
    subtracted. Unmatched frames remain neutral unless ``missing_anchor_policy``
    is ``error``.
    """

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    _validate_anchor_config(anchor_config)

    rows = normalize_candidate_columns(pd.DataFrame(candidates).copy()).reset_index(drop=True)
    sequence_ids = sorted(rows["sequence_id"].astype(str).unique().tolist()) if not rows.empty else []
    anchors = _normalize_anchor_estimates(initial_estimates, sequence_ids=sequence_ids)

    if rows.empty:
        return rows, anchors, _anchor_summary(rows, anchors, anchor_config=anchor_config)

    raw_score = core._candidate_scores(rows, config=mixture_config)
    sigma = core._candidate_sigmas(rows, config=mixture_config)
    rows["mixture_anchor_base_raw_score"] = raw_score.to_numpy(float)
    rows["mixture_anchor_sigma_m"] = sigma.to_numpy(float)
    rows["mixture_anchor_base_utility"] = np.nan

    temperature = max(float(mixture_config.temperature), 1.0e-12)
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        normalized = core._normalize_scores(
            raw_score.loc[frame.index].to_numpy(float),
            mode=mixture_config.score_normalization,
        )
        utility = (
            float(mixture_config.score_weight) * normalized / temperature
            - float(mixture_config.sigma_log_weight)
            * np.log(np.maximum(sigma.loc[frame.index].to_numpy(float), 1.0e-12))
        )
        rows.loc[frame.index, "mixture_anchor_base_utility"] = utility

    rows["mixture_anchor_matched"] = False
    rows["mixture_anchor_time_delta_s"] = np.nan
    rows["mixture_anchor_x_m"] = np.nan
    rows["mixture_anchor_y_m"] = np.nan
    rows["mixture_anchor_z_m"] = np.nan
    rows["mixture_anchor_distance_m"] = np.nan
    rows["mixture_anchor_cost"] = 0.0
    rows[ANCHOR_UTILITY_COLUMN] = rows["mixture_anchor_base_utility"].to_numpy(float)

    missing_frames: list[tuple[str, float]] = []
    for (sequence_id, time_s), frame in rows.groupby(
        ["sequence_id", "time_s"], sort=True, dropna=False
    ):
        anchor = _interpolate_anchor(
            anchors.loc[anchors["sequence_id"].astype(str) == str(sequence_id)],
            time_s=float(time_s),
            tolerance_s=float(anchor_config.anchor_time_tolerance_s),
        )
        if anchor is None:
            missing_frames.append((str(sequence_id), float(time_s)))
            continue
        anchor_xyz, time_delta_s = anchor
        positions = frame[["x_m", "y_m", "z_m"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(float)
        distance = np.linalg.norm(positions - anchor_xyz[None, :], axis=1)
        normalized_distance = distance / float(anchor_config.anchor_scale_m)
        anchor_cost = _huber_cost(
            normalized_distance,
            delta=float(anchor_config.anchor_huber_delta),
        )
        if float(anchor_config.anchor_cost_cap) > 0.0:
            anchor_cost = np.minimum(anchor_cost, float(anchor_config.anchor_cost_cap))
        utility = (
            rows.loc[frame.index, "mixture_anchor_base_utility"].to_numpy(float)
            - float(anchor_config.anchor_selection_weight) * anchor_cost
        )
        rows.loc[frame.index, "mixture_anchor_matched"] = True
        rows.loc[frame.index, "mixture_anchor_time_delta_s"] = float(time_delta_s)
        rows.loc[frame.index, "mixture_anchor_x_m"] = float(anchor_xyz[0])
        rows.loc[frame.index, "mixture_anchor_y_m"] = float(anchor_xyz[1])
        rows.loc[frame.index, "mixture_anchor_z_m"] = float(anchor_xyz[2])
        rows.loc[frame.index, "mixture_anchor_distance_m"] = distance
        rows.loc[frame.index, "mixture_anchor_cost"] = anchor_cost
        rows.loc[frame.index, ANCHOR_UTILITY_COLUMN] = utility

    if missing_frames and anchor_config.missing_anchor_policy == "error":
        example = ", ".join(f"{sequence}@{time_s:g}" for sequence, time_s in missing_frames[:5])
        raise ValueError(
            "missing anchor trajectory support for candidate frames: "
            f"{example}{' ...' if len(missing_frames) > 5 else ''}"
        )

    return rows, anchors, _anchor_summary(rows, anchors, anchor_config=anchor_config)


def select_anchor_posterior_mass_hypothesis_group_topk(
    candidates: pd.DataFrame,
    *,
    initial_estimates: pd.DataFrame,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select adaptive physical-hypothesis groups using anchor-conditioned unary."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()

    scored, anchors, anchor_summary = add_anchor_conditioned_selection_utility(
        candidates,
        initial_estimates,
        mixture_config=mixture_config,
        anchor_config=anchor_config,
    )
    selection_mixture_config = replace(
        mixture_config,
        score_column=ANCHOR_UTILITY_COLUMN,
        fallback_score_columns=(),
        score_normalization="none",
        score_weight=1.0,
        temperature=1.0,
        sigma_log_weight=0.0,
    )
    selected, summary = select_posterior_mass_hypothesis_group_topk(
        scored,
        mixture_config=selection_mixture_config,
        group_config=group_config,
        selection_config=selection_config,
    )
    summary = dict(summary)
    summary["schema"] = "raft-uav-mmuad-anchor-posterior-mass-group-topk-v1"
    summary["anchor_conditioning"] = anchor_summary
    summary["selection_mixture_config"] = asdict(selection_mixture_config)
    summary["truth_used_for_selection"] = False
    return selected, anchors, _jsonable(summary)


def run_anchor_posterior_mass_group_topk_candidate_mixture_map(
    candidates: pd.DataFrame,
    *,
    initial_estimates: pd.DataFrame,
    mixture_config: CandidateMixtureMapConfig | None = None,
    group_config: HypothesisGroupConfig | None = None,
    selection_config: PosteriorMassGroupTopKConfig | None = None,
    anchor_config: AnchorConditioningConfig | None = None,
    truth: pd.DataFrame | None = None,
) -> AnchorPosteriorMassGroupTopKCandidateMixtureResult:
    """Run anchor-conditioned adaptive group selection and grouped mixture-MAP."""

    mixture_config = mixture_config or CandidateMixtureMapConfig()
    group_config = group_config or HypothesisGroupConfig()
    selection_config = selection_config or PosteriorMassGroupTopKConfig()
    anchor_config = anchor_config or AnchorConditioningConfig()
    selected, anchors, summary = select_anchor_posterior_mass_hypothesis_group_topk(
        candidates,
        initial_estimates=initial_estimates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        anchor_config=anchor_config,
    )
    effective_mixture_config = mixture_config
    if int(selection_config.max_group_top_k) > 0:
        effective_mixture_config = replace(mixture_config, top_k=0)
    grouped = run_grouped_candidate_mixture_map(
        selected,
        mixture_config=effective_mixture_config,
        group_config=group_config,
        initial_estimates=anchors,
        truth=truth,
    )
    summary["final_mixture_config"] = asdict(effective_mixture_config)
    return AnchorPosteriorMassGroupTopKCandidateMixtureResult(
        scored_candidates=selected,
        selected_candidates=selected,
        grouped_result=grouped,
        selection_summary=_jsonable(summary),
    )


def write_anchor_posterior_mass_group_topk_outputs(
    result: AnchorPosteriorMassGroupTopKCandidateMixtureResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Write anchor-selection diagnostics and standard grouped mixture artifacts."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "mmuad_anchor_posterior_mass_group_topk_candidates.csv"
    summary_path = output / "mmuad_anchor_posterior_mass_group_topk_summary.json"
    result.selected_candidates.to_csv(selected_path, index=False)
    summary_path.write_text(
        json.dumps(_jsonable(result.selection_summary), indent=2), encoding="utf-8"
    )
    paths = write_grouped_candidate_mixture_outputs(result.grouped_result, output)
    paths["anchor_posterior_mass_group_topk_candidates_csv"] = selected_path
    paths["anchor_posterior_mass_group_topk_summary_json"] = summary_path
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m raft_uav.mmuad.candidate_mixture_group_anchor_mass_topk",
        description="condition adaptive MMUAD group selection on an initial trajectory",
    )
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--initial-estimates-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--min-group-top-k", type=int, default=3)
    parser.add_argument("--max-group-top-k", type=int, default=20)
    parser.add_argument("--target-posterior-mass", type=float, default=0.95)
    parser.add_argument("--posterior-temperature", type=float, default=1.0)
    parser.add_argument("--uniform-posterior-blend", type=float, default=0.02)
    parser.add_argument("--max-siblings-per-group", type=int, default=2)
    parser.add_argument("--group-score-mode", choices=GROUP_SCORE_MODES, default="max")
    parser.add_argument("--diversity-weight", type=float, default=0.5)
    parser.add_argument("--diversity-scale-m", type=float, default=5.0)
    parser.add_argument("--diversity-cap-m", type=float, default=30.0)
    parser.add_argument("--anchor-selection-weight", type=float, default=1.0)
    parser.add_argument("--anchor-scale-m", type=float, default=10.0)
    parser.add_argument("--anchor-huber-delta", type=float, default=1.0)
    parser.add_argument("--anchor-cost-cap", type=float, default=4.0)
    parser.add_argument("--anchor-time-tolerance-s", type=float, default=0.5)
    parser.add_argument(
        "--missing-anchor-policy", choices=MISSING_ANCHOR_POLICIES, default="neutral"
    )
    parser.add_argument("--row-top-k-when-disabled", type=int, default=20)
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--fallback-score-column", action="append", default=[])
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--default-sigma-m", type=float, default=10.0)
    parser.add_argument("--sigma-min-m", type=float, default=1.0)
    parser.add_argument("--sigma-max-m", type=float, default=30.0)
    parser.add_argument(
        "--score-normalization", choices=SCORE_NORMALIZATION_CHOICES, default="minmax"
    )
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--sigma-log-weight", type=float, default=3.0)
    parser.add_argument("--loss", choices=LOSS_CHOICES, default="huber")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--tolerance-m", type=float, default=1.0e-3)
    parser.add_argument("--uniform-weight-floor", type=float, default=0.0)
    parser.add_argument("--branch-balance", type=float, default=0.0)
    parser.add_argument("--source-balance", type=float, default=0.0)
    parser.add_argument("--responsibility-floor", type=float, default=0.0)
    parser.add_argument(
        "--initialization", choices=INITIALIZATION_CHOICES, default="uncertainty-top1"
    )
    parser.add_argument("--hypothesis-group-column")
    parser.add_argument("--hypothesis-group-correction-strength", type=float, default=1.0)
    parser.add_argument(
        "--missing-hypothesis-group-policy", choices=("unique", "error"), default="unique"
    )
    args = parser.parse_args(argv)

    fallback = tuple(args.fallback_score_column) or ("ranker_score", "confidence")
    mixture_config = CandidateMixtureMapConfig(
        top_k=args.row_top_k_when_disabled,
        score_column=args.score_column,
        fallback_score_columns=fallback,
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
        anchor_weight=args.anchor_weight,
        iterations=args.iterations,
        tolerance_m=args.tolerance_m,
        uniform_weight_floor=args.uniform_weight_floor,
        branch_balance=args.branch_balance,
        source_balance=args.source_balance,
        responsibility_floor=args.responsibility_floor,
        initialization=args.initialization,
    )
    group_config = HypothesisGroupConfig(
        group_column=args.hypothesis_group_column,
        correction_strength=args.hypothesis_group_correction_strength,
        missing_group_policy=args.missing_hypothesis_group_policy,
    )
    selection_config = PosteriorMassGroupTopKConfig(
        min_group_top_k=args.min_group_top_k,
        max_group_top_k=args.max_group_top_k,
        target_posterior_mass=args.target_posterior_mass,
        posterior_temperature=args.posterior_temperature,
        uniform_posterior_blend=args.uniform_posterior_blend,
        max_siblings_per_group=args.max_siblings_per_group,
        group_score_mode=args.group_score_mode,
        diversity_weight=args.diversity_weight,
        diversity_scale_m=args.diversity_scale_m,
        diversity_cap_m=args.diversity_cap_m,
    )
    anchor_config = AnchorConditioningConfig(
        anchor_selection_weight=args.anchor_selection_weight,
        anchor_scale_m=args.anchor_scale_m,
        anchor_huber_delta=args.anchor_huber_delta,
        anchor_cost_cap=args.anchor_cost_cap,
        anchor_time_tolerance_s=args.anchor_time_tolerance_s,
        missing_anchor_policy=args.missing_anchor_policy,
    )
    candidates = load_candidate_file(args.candidates_csv).rows
    initial_estimates = read_estimate_csv(args.initial_estimates_csv)
    truth = (
        None
        if args.truth_csv is None
        else load_evaluation_truth_file(args.truth_csv).rows
    )
    result = run_anchor_posterior_mass_group_topk_candidate_mixture_map(
        candidates,
        initial_estimates=initial_estimates,
        mixture_config=mixture_config,
        group_config=group_config,
        selection_config=selection_config,
        anchor_config=anchor_config,
        truth=truth,
    )
    paths = write_anchor_posterior_mass_group_topk_outputs(result, args.output_dir)
    print("mmuad_anchor_posterior_mass_group_topk=ok")
    print(f"input_candidate_rows={len(candidates)}")
    print(f"selected_candidate_rows={len(result.selected_candidates)}")
    pooled = result.grouped_result.mixture_result.summary.get("metrics", {}).get(
        "pooled", {}
    )
    if pooled.get("rmse_3d_m") is not None:
        print(f"rmse_3d_m={pooled['rmse_3d_m']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _normalize_anchor_estimates(
    initial_estimates: pd.DataFrame,
    *,
    sequence_ids: list[str],
) -> pd.DataFrame:
    rows = pd.DataFrame(initial_estimates).copy()
    rows.columns = [str(column).strip() for column in rows.columns]
    sequence_column = _resolve_alias(rows, _SEQUENCE_ALIASES)
    time_column = _resolve_alias(rows, _TIME_ALIASES)
    x_column = _resolve_alias(rows, _X_ALIASES)
    y_column = _resolve_alias(rows, _Y_ALIASES)
    z_column = _resolve_alias(rows, _Z_ALIASES)
    missing = [
        name
        for name, column in (
            ("time", time_column),
            ("x", x_column),
            ("y", y_column),
            ("z", z_column),
        )
        if column is None
    ]
    if missing:
        raise ValueError(f"initial estimates missing required columns: {missing}")

    normalized = pd.DataFrame(
        {
            "time_s": pd.to_numeric(rows[time_column], errors="coerce"),
            "state_x_m": pd.to_numeric(rows[x_column], errors="coerce"),
            "state_y_m": pd.to_numeric(rows[y_column], errors="coerce"),
            "state_z_m": pd.to_numeric(rows[z_column], errors="coerce"),
        }
    )
    if sequence_column is None:
        parts = []
        for sequence_id in sequence_ids:
            part = normalized.copy()
            part.insert(0, "sequence_id", str(sequence_id))
            parts.append(part)
        normalized = pd.concat(parts, ignore_index=True) if parts else normalized.assign(
            sequence_id="default"
        )
    else:
        normalized.insert(0, "sequence_id", rows[sequence_column].astype(str).str.strip())

    numeric_columns = ["time_s", "state_x_m", "state_y_m", "state_z_m"]
    finite = np.isfinite(normalized[numeric_columns].to_numpy(float)).all(axis=1)
    normalized = normalized.loc[finite].copy()
    if normalized.empty:
        raise ValueError("initial estimates contain no finite trajectory rows")
    normalized = (
        normalized.groupby(["sequence_id", "time_s"], as_index=False, sort=True)[
            ["state_x_m", "state_y_m", "state_z_m"]
        ]
        .mean()
        .sort_values(["sequence_id", "time_s"], kind="mergesort")
        .reset_index(drop=True)
    )
    return normalized


def _resolve_alias(rows: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    columns = {str(column).strip().lower(): str(column) for column in rows.columns}
    for alias in aliases:
        resolved = columns.get(str(alias).strip().lower())
        if resolved is not None:
            return resolved
    return None


def _interpolate_anchor(
    anchors: pd.DataFrame,
    *,
    time_s: float,
    tolerance_s: float,
) -> tuple[np.ndarray, float] | None:
    if anchors.empty:
        return None
    ordered = anchors.sort_values("time_s", kind="mergesort")
    times = ordered["time_s"].to_numpy(float)
    nearest_delta = float(np.min(np.abs(times - float(time_s))))
    if nearest_delta > float(tolerance_s):
        return None
    xyz = ordered[["state_x_m", "state_y_m", "state_z_m"]].to_numpy(float)
    if len(times) == 1:
        return xyz[0], nearest_delta
    interpolated = np.asarray(
        [np.interp(float(time_s), times, xyz[:, axis]) for axis in range(3)],
        dtype=float,
    )
    return interpolated, nearest_delta


def _huber_cost(values: np.ndarray, *, delta: float) -> np.ndarray:
    residual = np.abs(np.asarray(values, dtype=float))
    return np.where(
        residual <= float(delta),
        0.5 * residual**2,
        float(delta) * (residual - 0.5 * float(delta)),
    )


def _anchor_summary(
    rows: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    anchor_config: AnchorConditioningConfig,
) -> dict[str, Any]:
    matched = (
        rows["mixture_anchor_matched"].astype(bool)
        if "mixture_anchor_matched" in rows.columns
        else pd.Series(False, index=rows.index)
    )
    distance = pd.to_numeric(
        rows.get("mixture_anchor_distance_m", pd.Series(dtype=float)), errors="coerce"
    )
    distance = distance.loc[np.isfinite(distance)]
    frame_rows = (
        rows[["sequence_id", "time_s", "mixture_anchor_matched"]].drop_duplicates()
        if {"sequence_id", "time_s", "mixture_anchor_matched"}.issubset(rows.columns)
        else pd.DataFrame()
    )
    matched_frames = (
        int(frame_rows["mixture_anchor_matched"].astype(bool).sum())
        if not frame_rows.empty
        else 0
    )
    return {
        "config": asdict(anchor_config),
        "anchor_rows": int(len(anchors)),
        "candidate_rows": int(len(rows)),
        "matched_candidate_rows": int(matched.sum()),
        "matched_frame_count": matched_frames,
        "frame_count": int(len(frame_rows)),
        "matched_frame_fraction": (
            float(matched_frames / len(frame_rows)) if len(frame_rows) else 0.0
        ),
        "anchor_distance_mean_m": float(distance.mean()) if len(distance) else None,
        "anchor_distance_p95_m": float(distance.quantile(0.95)) if len(distance) else None,
    }


def _validate_anchor_config(config: AnchorConditioningConfig) -> None:
    for name in (
        "anchor_selection_weight",
        "anchor_scale_m",
        "anchor_huber_delta",
        "anchor_cost_cap",
        "anchor_time_tolerance_s",
    ):
        value = float(getattr(config, name))
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if float(config.anchor_selection_weight) < 0.0:
        raise ValueError("anchor_selection_weight must be non-negative")
    if float(config.anchor_scale_m) <= 0.0:
        raise ValueError("anchor_scale_m must be positive")
    if float(config.anchor_huber_delta) <= 0.0:
        raise ValueError("anchor_huber_delta must be positive")
    if float(config.anchor_cost_cap) < 0.0:
        raise ValueError("anchor_cost_cap must be non-negative")
    if float(config.anchor_time_tolerance_s) < 0.0:
        raise ValueError("anchor_time_tolerance_s must be non-negative")
    if config.missing_anchor_policy not in MISSING_ANCHOR_POLICIES:
        raise ValueError(
            f"unsupported missing_anchor_policy={config.missing_anchor_policy!r}"
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
