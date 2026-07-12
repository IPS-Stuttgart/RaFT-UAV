"""Train-only selection for agreement-adaptive MMUAD pair-state priors.

The pair-state forward-backward model can recover coherent candidates buried by
the framewise ranker, but its blend controls must be frozen without using
public-validation or hidden-test truth. This module evaluates candidate-posterior
blend settings across training sequences, ranks them by a mean/tail risk score,
and writes one frozen configuration plus selected candidate posteriors.

The expensive local and pair-state posteriors are computed once per sequence.
Grid rows only reblend those fixed posteriors before the learned-sigma Huber
candidate-mixture smoother is evaluated.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_mixture_map import (
    CandidateMixtureMapConfig,
    run_candidate_mixture_map,
)
from raft_uav.mmuad.candidate_pair_forward_backward import (
    CandidatePairForwardBackwardConfig,
    pair_forward_backward_summary,
)
from raft_uav.mmuad.candidate_pair_forward_backward_agreement_adaptive import (
    AgreementAdaptivePairBlendConfig,
    attach_agreement_adaptive_pair_prior,
    blend_candidate_posteriors,
)
from raft_uav.mmuad.evaluator import load_evaluation_truth_file
from raft_uav.mmuad.io import load_candidate_file
from raft_uav.mmuad.schema import (
    CandidateFrame,
    normalize_candidate_columns,
    normalize_truth_columns,
)

_ALLOWED_SELECTION_METRICS = (
    "mse_3d_m",
    "mean_3d_m",
    "rmse_3d_m",
    "p95_3d_m",
    "max_3d_m",
)
_DEFAULT_MIN_PAIR_WEIGHTS = (0.0, 0.1)
_DEFAULT_MAX_PAIR_WEIGHTS = (0.75, 1.0)
_DEFAULT_ENTROPY_POWERS = (1.0, 2.0)
_DEFAULT_AGREEMENT_POWERS = (0.5, 1.0)
_DEFAULT_AGREEMENT_FLOORS = (0.0, 0.1)
_SEED_SCORE_COLUMN = "candidate_pair_forward_backward_agreement_cv_seed_score"

FOLD_CSV = "mmuad_agreement_pair_cv_folds.csv"
AGGREGATE_CSV = "mmuad_agreement_pair_cv_aggregate.csv"
SELECTED_CONFIG_JSON = "mmuad_agreement_pair_cv_selected_config.json"
SELECTED_CANDIDATES_CSV = "mmuad_agreement_pair_cv_selected_candidates.csv"
SELECTED_SUMMARY_JSON = "mmuad_agreement_pair_cv_selected_summary.json"


@dataclass(frozen=True)
class AgreementPairCVConfig:
    """Risk-aware training-sequence selection controls."""

    selection_metric: str = "mse_3d_m"
    risk_aversion: float = 0.25
    tail_quantile: float = 0.9


def select_agreement_pair_config_by_sequence_cv(
    candidates: CandidateFrame | pd.DataFrame,
    truth: pd.DataFrame,
    *,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    mixture_config: CandidateMixtureMapConfig | None = None,
    cv_config: AgreementPairCVConfig | None = None,
    min_pair_weights: Sequence[float] = _DEFAULT_MIN_PAIR_WEIGHTS,
    max_pair_weights: Sequence[float] = _DEFAULT_MAX_PAIR_WEIGHTS,
    entropy_powers: Sequence[float] = _DEFAULT_ENTROPY_POWERS,
    agreement_powers: Sequence[float] = _DEFAULT_AGREEMENT_POWERS,
    agreement_floors: Sequence[float] = _DEFAULT_AGREEMENT_FLOORS,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, CandidateFrame]:
    """Select one frozen blend using only supplied training sequences and truth."""

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    mixture_cfg = mixture_config or CandidateMixtureMapConfig()
    cv_cfg = cv_config or AgreementPairCVConfig()
    _validate_cv_config(cv_cfg)

    rows = _candidate_rows(candidates)
    truth_rows = normalize_truth_columns(pd.DataFrame(truth).copy())
    if rows.empty or truth_rows.empty:
        raise ValueError("candidate and train-truth rows must not be empty")
    sequences = sorted(
        set(rows["sequence_id"].astype(str))
        & set(truth_rows["sequence_id"].astype(str))
    )
    if len(sequences) < 2:
        raise ValueError("agreement-pair CV requires at least two shared train sequences")

    prepared: dict[str, CandidateFrame] = {}
    truth_by_sequence: dict[str, pd.DataFrame] = {}
    for sequence_id in sequences:
        sequence_rows = rows.loc[rows["sequence_id"].astype(str) == sequence_id]
        prepared[sequence_id] = prepare_agreement_pair_posteriors(
            sequence_rows,
            pair_config=pair_cfg,
        )
        truth_by_sequence[sequence_id] = truth_rows.loc[
            truth_rows["sequence_id"].astype(str) == sequence_id
        ]

    fold_records: list[dict[str, Any]] = []
    blend_configs = _blend_grid(
        min_pair_weights=min_pair_weights,
        max_pair_weights=max_pair_weights,
        entropy_powers=entropy_powers,
        agreement_powers=agreement_powers,
        agreement_floors=agreement_floors,
    )
    for blend_cfg in blend_configs:
        for sequence_id in sequences:
            augmented = reblend_prepared_agreement_pair_posteriors(
                prepared[sequence_id],
                pair_config=pair_cfg,
                blend_config=blend_cfg,
            )
            result = run_candidate_mixture_map(
                augmented.rows,
                config=_mixture_config_for_blend(
                    mixture_cfg,
                    pair_config=pair_cfg,
                    blend_config=blend_cfg,
                ),
                truth=truth_by_sequence[sequence_id],
            )
            metrics = dict(result.summary.get("metrics", {}).get("pooled", {}))
            rmse = _finite_or_none(metrics.get("rmse_3d_m"))
            metrics["mse_3d_m"] = None if rmse is None else rmse**2
            fold_records.append(
                {
                    "grid_label": _blend_label(blend_cfg),
                    "holdout_sequence_id": sequence_id,
                    **_blend_record(blend_cfg),
                    **{name: metrics.get(name) for name in _ALLOWED_SELECTION_METRICS},
                    "metric_count": int(metrics.get("count", 0)),
                }
            )

    folds = pd.DataFrame.from_records(fold_records)
    aggregate = aggregate_agreement_pair_cv_folds(
        folds,
        cv_config=cv_cfg,
        expected_sequence_count=len(sequences),
    )
    if aggregate.empty or not bool(aggregate.iloc[0]["eligible"]):
        raise ValueError("agreement-pair CV did not produce an eligible configuration")

    selected_row = aggregate.iloc[0]
    selected_blend = AgreementAdaptivePairBlendConfig(
        min_pair_weight=float(selected_row["min_pair_weight"]),
        max_pair_weight=float(selected_row["max_pair_weight"]),
        entropy_power=float(selected_row["entropy_power"]),
        agreement_power=float(selected_row["agreement_power"]),
        agreement_floor=float(selected_row["agreement_floor"]),
    )
    prepared_all = CandidateFrame(
        pd.concat(
            [prepared[sequence_id].rows for sequence_id in sequences],
            ignore_index=True,
        )
    )
    selected_candidates = reblend_prepared_agreement_pair_posteriors(
        prepared_all,
        pair_config=pair_cfg,
        blend_config=selected_blend,
    )
    selected_mixture = _mixture_config_for_blend(
        mixture_cfg,
        pair_config=pair_cfg,
        blend_config=selected_blend,
    )
    selected_config = {
        "schema": "raft-uav-mmuad-agreement-pair-train-cv-v1",
        "selection_protocol": "train-sequence-cv-aggregate",
        "sequence_ids": sequences,
        "sequence_count": len(sequences),
        "cv_config": asdict(cv_cfg),
        "pair_config": asdict(pair_cfg),
        "mixture_config": asdict(selected_mixture),
        "blend_config": asdict(selected_blend),
        "selected_grid_label": str(selected_row["grid_label"]),
        "selected_metric_mean": _finite_or_none(
            selected_row.get(f"{cv_cfg.selection_metric}_mean")
        ),
        "selected_metric_tail": _finite_or_none(
            selected_row.get(f"{cv_cfg.selection_metric}_tail")
        ),
        "selected_risk_score": _finite_or_none(selected_row.get("risk_score")),
        "truth_used_for_candidate_prior": False,
        "truth_used_for_train_cv_selection": True,
    }
    return selected_config, folds, aggregate, selected_candidates


def prepare_agreement_pair_posteriors(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
) -> CandidateFrame:
    """Compute expensive local and pair-state posteriors once."""

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    seed_blend = AgreementAdaptivePairBlendConfig(
        min_pair_weight=0.0,
        max_pair_weight=0.0,
        output_score_column=_SEED_SCORE_COLUMN,
    )
    return attach_agreement_adaptive_pair_prior(
        candidates,
        pair_config=pair_cfg,
        blend_config=seed_blend,
    )


def reblend_prepared_agreement_pair_posteriors(
    candidates: CandidateFrame | pd.DataFrame,
    *,
    pair_config: CandidatePairForwardBackwardConfig | None = None,
    blend_config: AgreementAdaptivePairBlendConfig | None = None,
) -> CandidateFrame:
    """Reblend fixed local/pair posteriors without rerunning pair inference."""

    pair_cfg = pair_config or CandidatePairForwardBackwardConfig()
    blend_cfg = blend_config or AgreementAdaptivePairBlendConfig()
    rows = _candidate_rows(candidates)
    if rows.empty:
        return CandidateFrame(rows)

    local_column = "candidate_pair_forward_backward_agreement_local_posterior"
    missing = [
        column
        for column in (local_column, pair_cfg.output_score_column)
        if column not in rows.columns
    ]
    if missing:
        raise ValueError(
            "prepared agreement-pair rows missing columns: "
            + ", ".join(sorted(missing))
        )

    out = rows.copy()
    for _, frame in out.groupby(["sequence_id", "time_s"], sort=False):
        indices = frame.index
        local = pd.to_numeric(frame[local_column], errors="coerce").to_numpy(float)
        pair = pd.to_numeric(
            frame[pair_cfg.output_score_column],
            errors="coerce",
        ).to_numpy(float)
        blended, diagnostics = blend_candidate_posteriors(
            local,
            pair,
            config=blend_cfg,
        )
        out.loc[indices, blend_cfg.output_score_column] = blended
        out.loc[
            indices,
            "candidate_pair_forward_backward_agreement_adaptive_rank",
        ] = _descending_ranks(blended)
        for name, value in diagnostics.items():
            out.loc[
                indices,
                f"candidate_pair_forward_backward_agreement_{name}",
            ] = value
    return CandidateFrame(normalize_candidate_columns(out))


def aggregate_agreement_pair_cv_folds(
    fold_summary: pd.DataFrame,
    *,
    cv_config: AgreementPairCVConfig | None = None,
    expected_sequence_count: int | None = None,
) -> pd.DataFrame:
    """Aggregate per-sequence metrics and rank configs by mean/tail risk."""

    cv_cfg = cv_config or AgreementPairCVConfig()
    _validate_cv_config(cv_cfg)
    rows = pd.DataFrame(fold_summary).copy()
    if rows.empty:
        return rows
    metric = cv_cfg.selection_metric
    if metric not in rows.columns:
        raise ValueError(f"fold summary missing selection metric {metric!r}")

    config_columns = [
        "grid_label",
        "min_pair_weight",
        "max_pair_weight",
        "entropy_power",
        "agreement_power",
        "agreement_floor",
    ]
    records: list[dict[str, Any]] = []
    for _, group in rows.groupby(config_columns, sort=False, dropna=False):
        values = pd.to_numeric(group[metric], errors="coerce")
        finite = values[np.isfinite(values.to_numpy(float))]
        expected = (
            int(expected_sequence_count)
            if expected_sequence_count is not None
            else int(group["holdout_sequence_id"].astype(str).nunique())
        )
        mean = float(finite.mean()) if not finite.empty else float("nan")
        tail = (
            float(finite.quantile(cv_cfg.tail_quantile))
            if not finite.empty
            else float("nan")
        )
        record = {column: group.iloc[0][column] for column in config_columns}
        record.update(
            {
                "fold_count": int(len(group)),
                "valid_fold_count": int(len(finite)),
                "eligible": bool(len(finite) == expected and expected > 0),
                f"{metric}_mean": mean,
                f"{metric}_std": (
                    float(finite.std(ddof=0)) if not finite.empty else float("nan")
                ),
                f"{metric}_tail": tail,
                f"{metric}_worst": (
                    float(finite.max()) if not finite.empty else float("nan")
                ),
            }
        )
        record["risk_score"] = (
            (1.0 - cv_cfg.risk_aversion) * mean + cv_cfg.risk_aversion * tail
            if record["eligible"] and np.isfinite(mean) and np.isfinite(tail)
            else float("inf")
        )
        records.append(record)

    aggregate = pd.DataFrame.from_records(records)
    return aggregate.sort_values(
        ["eligible", "risk_score", f"{metric}_mean", f"{metric}_std", "grid_label"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


def write_agreement_pair_cv_outputs(
    *,
    selected_config: dict[str, Any],
    fold_summary: pd.DataFrame,
    aggregate_summary: pd.DataFrame,
    selected_candidates: CandidateFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Write fold metrics, selected config, and inference-ready candidates."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold_csv": output / FOLD_CSV,
        "aggregate_csv": output / AGGREGATE_CSV,
        "selected_config_json": output / SELECTED_CONFIG_JSON,
        "selected_candidates_csv": output / SELECTED_CANDIDATES_CSV,
        "selected_summary_json": output / SELECTED_SUMMARY_JSON,
    }
    fold_summary.to_csv(paths["fold_csv"], index=False)
    aggregate_summary.to_csv(paths["aggregate_csv"], index=False)
    selected_candidates.rows.to_csv(paths["selected_candidates_csv"], index=False)
    paths["selected_config_json"].write_text(
        json.dumps(_jsonable(selected_config), indent=2),
        encoding="utf-8",
    )
    pair_score_column = str(selected_config["pair_config"]["output_score_column"])
    output_score_column = str(selected_config["blend_config"]["output_score_column"])
    selected_summary = {
        "schema": "raft-uav-mmuad-agreement-pair-selected-candidates-v1",
        "pair_summary": pair_forward_backward_summary(
            selected_candidates,
            score_column=pair_score_column,
        ),
        "posterior_sum_error_max": _posterior_sum_error(
            selected_candidates.rows,
            output_score_column,
        ),
        "selected_config_json": str(paths["selected_config_json"]),
    }
    paths["selected_summary_json"].write_text(
        json.dumps(_jsonable(selected_summary), indent=2),
        encoding="utf-8",
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mmuad-candidate-pair-forward-backward-agreement-cv",
        description="select an agreement-adaptive pair prior on MMUAD train sequences",
    )
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-column", default="candidate_reservoir_grid_score")
    parser.add_argument("--sigma-column", default="predicted_sigma_m")
    parser.add_argument("--transition-distance-std-m", type=float, default=2.0)
    parser.add_argument("--transition-speed-std-mps", type=float, default=15.0)
    parser.add_argument("--acceleration-std-mps2", type=float, default=20.0)
    parser.add_argument("--min-pair-weight", type=float, action="append")
    parser.add_argument("--max-pair-weight", type=float, action="append")
    parser.add_argument("--entropy-power", type=float, action="append")
    parser.add_argument("--agreement-power", type=float, action="append")
    parser.add_argument("--agreement-floor", type=float, action="append")
    parser.add_argument(
        "--selection-metric",
        choices=_ALLOWED_SELECTION_METRICS,
        default="mse_3d_m",
    )
    parser.add_argument("--risk-aversion", type=float, default=0.25)
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--mixture-top-k", type=int, default=20)
    parser.add_argument("--mixture-smoothness-weight", type=float, default=7200.0)
    parser.add_argument("--mixture-huber-delta", type=float, default=1.0)
    parser.add_argument("--mixture-iterations", type=int, default=5)
    args = parser.parse_args(argv)

    pair_cfg = CandidatePairForwardBackwardConfig(
        score_column=str(args.score_column),
        sigma_column=str(args.sigma_column),
        transition_distance_std_m=float(args.transition_distance_std_m),
        transition_speed_std_mps=float(args.transition_speed_std_mps),
        acceleration_std_mps2=float(args.acceleration_std_mps2),
    )
    mixture_cfg = CandidateMixtureMapConfig(
        top_k=int(args.mixture_top_k),
        sigma_column=str(args.sigma_column),
        score_normalization="none",
        loss="huber",
        huber_delta=float(args.mixture_huber_delta),
        smoothness_weight=float(args.mixture_smoothness_weight),
        iterations=int(args.mixture_iterations),
    )
    selected, folds, aggregate, selected_candidates = (
        select_agreement_pair_config_by_sequence_cv(
            load_candidate_file(args.candidate_csv),
            load_evaluation_truth_file(args.truth_csv).rows,
            pair_config=pair_cfg,
            mixture_config=mixture_cfg,
            cv_config=AgreementPairCVConfig(
                selection_metric=str(args.selection_metric),
                risk_aversion=float(args.risk_aversion),
                tail_quantile=float(args.tail_quantile),
            ),
            min_pair_weights=_values_or_default(
                args.min_pair_weight,
                _DEFAULT_MIN_PAIR_WEIGHTS,
            ),
            max_pair_weights=_values_or_default(
                args.max_pair_weight,
                _DEFAULT_MAX_PAIR_WEIGHTS,
            ),
            entropy_powers=_values_or_default(
                args.entropy_power,
                _DEFAULT_ENTROPY_POWERS,
            ),
            agreement_powers=_values_or_default(
                args.agreement_power,
                _DEFAULT_AGREEMENT_POWERS,
            ),
            agreement_floors=_values_or_default(
                args.agreement_floor,
                _DEFAULT_AGREEMENT_FLOORS,
            ),
        )
    )
    paths = write_agreement_pair_cv_outputs(
        selected_config=selected,
        fold_summary=folds,
        aggregate_summary=aggregate,
        selected_candidates=selected_candidates,
        output_dir=args.output_dir,
    )
    print("mmuad_agreement_pair_cv=ok")
    print(f"selected_grid_label={selected['selected_grid_label']}")
    print(f"selected_risk_score={selected['selected_risk_score']}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


def _mixture_config_for_blend(
    config: CandidateMixtureMapConfig,
    *,
    pair_config: CandidatePairForwardBackwardConfig,
    blend_config: AgreementAdaptivePairBlendConfig,
) -> CandidateMixtureMapConfig:
    payload = asdict(config)
    payload["score_column"] = blend_config.output_score_column
    payload["fallback_score_columns"] = (
        pair_config.output_score_column,
        pair_config.score_column,
        *pair_config.fallback_score_columns,
    )
    payload["score_normalization"] = "none"
    return CandidateMixtureMapConfig(**payload)


def _blend_grid(
    *,
    min_pair_weights: Sequence[float],
    max_pair_weights: Sequence[float],
    entropy_powers: Sequence[float],
    agreement_powers: Sequence[float],
    agreement_floors: Sequence[float],
) -> list[AgreementAdaptivePairBlendConfig]:
    grids = [
        _finite_values(min_pair_weights, name="min_pair_weights"),
        _finite_values(max_pair_weights, name="max_pair_weights"),
        _finite_values(entropy_powers, name="entropy_powers"),
        _finite_values(agreement_powers, name="agreement_powers"),
        _finite_values(agreement_floors, name="agreement_floors"),
    ]
    configs = []
    for minimum, maximum, entropy, agreement, floor in product(*grids):
        if not 0.0 <= minimum <= maximum <= 1.0:
            continue
        if entropy <= 0.0 or agreement <= 0.0 or not 0.0 <= floor <= 1.0:
            continue
        configs.append(
            AgreementAdaptivePairBlendConfig(
                min_pair_weight=minimum,
                max_pair_weight=maximum,
                entropy_power=entropy,
                agreement_power=agreement,
                agreement_floor=floor,
            )
        )
    if not configs:
        raise ValueError("agreement-pair grid contains no valid configurations")
    return configs


def _blend_label(config: AgreementAdaptivePairBlendConfig) -> str:
    return (
        f"min{_token(config.min_pair_weight)}"
        f"_max{_token(config.max_pair_weight)}"
        f"_entropy{_token(config.entropy_power)}"
        f"_agreement{_token(config.agreement_power)}"
        f"_floor{_token(config.agreement_floor)}"
    )


def _blend_record(config: AgreementAdaptivePairBlendConfig) -> dict[str, float]:
    return {
        "min_pair_weight": float(config.min_pair_weight),
        "max_pair_weight": float(config.max_pair_weight),
        "entropy_power": float(config.entropy_power),
        "agreement_power": float(config.agreement_power),
        "agreement_floor": float(config.agreement_floor),
    }


def _candidate_rows(candidates: CandidateFrame | pd.DataFrame) -> pd.DataFrame:
    if isinstance(candidates, CandidateFrame):
        return normalize_candidate_columns(candidates.rows.copy())
    return normalize_candidate_columns(pd.DataFrame(candidates).copy())


def _descending_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(values, dtype=float), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1, dtype=int)
    return ranks


def _posterior_sum_error(rows: pd.DataFrame, score_column: str) -> float:
    sums = (
        rows.assign(_score=pd.to_numeric(rows[score_column], errors="coerce"))
        .groupby(["sequence_id", "time_s"], sort=False)["_score"]
        .sum()
    )
    return float(np.max(np.abs(sums.to_numpy(float) - 1.0))) if len(sums) else 0.0


def _validate_cv_config(config: AgreementPairCVConfig) -> None:
    if config.selection_metric not in _ALLOWED_SELECTION_METRICS:
        raise ValueError(f"unsupported selection metric {config.selection_metric!r}")
    if not 0.0 <= float(config.risk_aversion) <= 1.0:
        raise ValueError("risk_aversion must be within [0, 1]")
    if not 0.0 <= float(config.tail_quantile) <= 1.0:
        raise ValueError("tail_quantile must be within [0, 1]")


def _finite_values(values: Iterable[float], *, name: str) -> tuple[float, ...]:
    result = tuple(dict.fromkeys(float(value) for value in values))
    if not result or not np.isfinite(np.asarray(result, dtype=float)).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _values_or_default(
    values: Sequence[float] | None,
    default: Sequence[float],
) -> tuple[float, ...]:
    return tuple(default if not values else values)


def _token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


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
