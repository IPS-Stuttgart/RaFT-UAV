"""CI-style acceptance gate for paper-parity grid summaries.

The paper-parity grid ranks reproduction candidates, but ranking alone does not
protect downstream result runs from silently using a mismatched input protocol.
This module reads a ``raft-uav-paper-parity-grid`` summary CSV, selects the best
candidate, and exits non-zero when the candidate is still outside configured
Table-II count/error tolerances.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_REQUIRED_COUNT_DELTA_COLUMNS = (
    "rf_raw_count_delta",
    "rf_after_nis_count_delta",
    "radar_raw_count_delta",
    "radar_after_nis_count_delta",
    "kf_all_steps_count_delta",
    "kf_updated_count_delta",
    "kf_coasted_count_delta",
)
DEFAULT_MAX_KF_MEAN_ABS_DELTA_M = 5.0


@dataclass(frozen=True)
class PaperParityGateConfig:
    """Thresholds used to accept or reject a paper-parity grid summary."""

    max_count_abs_delta_total: int = 0
    max_stage_count_abs_delta: int = 0
    max_kf_mean_abs_delta_m: float | None = DEFAULT_MAX_KF_MEAN_ABS_DELTA_M
    max_paper_parity_score: float | None = None
    require_successful_best: bool = True
    required_count_delta_columns: tuple[str, ...] = DEFAULT_REQUIRED_COUNT_DELTA_COLUMNS


@dataclass(frozen=True)
class PaperParityGateResult:
    """Result returned by the paper-parity gate."""

    passed: bool
    reasons: tuple[str, ...]
    best_rank: int | None
    best_row: Mapping[str, Any]
    thresholds: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload."""

        return {
            "passed": bool(self.passed),
            "reasons": list(self.reasons),
            "best_rank": self.best_rank,
            "best_row": _jsonable_mapping(self.best_row),
            "thresholds": _jsonable_mapping(self.thresholds),
        }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the paper-parity acceptance gate from the command line."""

    parser = argparse.ArgumentParser(
        prog="raft-uav-paper-parity-gate",
        description=(
            "fail unless the best paper-parity grid row satisfies configured "
            "Table-II count/error tolerances"
        ),
    )
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-count-abs-delta-total", type=int, default=0)
    parser.add_argument("--max-stage-count-abs-delta", type=int, default=0)
    parser.add_argument(
        "--max-kf-mean-abs-delta-m",
        type=float,
        default=DEFAULT_MAX_KF_MEAN_ABS_DELTA_M,
        help="maximum allowed |KF all steps mean error delta|; use --no-kf-mean-check to skip",
    )
    parser.add_argument(
        "--no-kf-mean-check",
        action="store_true",
        help="accept based on counts/score only, without checking KF mean error delta",
    )
    parser.add_argument(
        "--max-paper-parity-score",
        type=float,
        default=None,
        help="optional maximum paper_parity_score for the best candidate",
    )
    parser.add_argument(
        "--allow-failed-best",
        action="store_true",
        help="do not reject a best row whose failed column is true",
    )
    parser.add_argument(
        "--required-count-delta-column",
        action="append",
        default=None,
        help=(
            "count-delta column that must be present and within "
            "--max-stage-count-abs-delta; repeat or pass comma-separated values"
        ),
    )
    parser.add_argument(
        "--no-required-count-columns",
        action="store_true",
        help="only check count_abs_delta_total, not individual stage count columns",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.no_required_count_columns:
        required_columns: tuple[str, ...] = ()
    else:
        required_columns = _parse_required_columns(args.required_count_delta_column)

    config = PaperParityGateConfig(
        max_count_abs_delta_total=args.max_count_abs_delta_total,
        max_stage_count_abs_delta=args.max_stage_count_abs_delta,
        max_kf_mean_abs_delta_m=None
        if args.no_kf_mean_check
        else args.max_kf_mean_abs_delta_m,
        max_paper_parity_score=args.max_paper_parity_score,
        require_successful_best=not args.allow_failed_best,
        required_count_delta_columns=required_columns,
    )
    result = gate_paper_parity_summary_csv(args.summary_csv, config=config)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    if not args.quiet:
        _print_gate_result(result)
    return 0 if result.passed else 1


def gate_paper_parity_summary_csv(
    summary_csv: Path,
    *,
    config: PaperParityGateConfig = PaperParityGateConfig(),
) -> PaperParityGateResult:
    """Load ``summary_csv`` and evaluate it with ``config``."""

    return evaluate_paper_parity_gate(pd.read_csv(summary_csv), config=config)


def evaluate_paper_parity_gate(
    summary: pd.DataFrame,
    *,
    config: PaperParityGateConfig = PaperParityGateConfig(),
) -> PaperParityGateResult:
    """Return pass/fail status for a paper-parity grid summary dataframe."""

    if summary.empty:
        return PaperParityGateResult(
            passed=False,
            reasons=("paper-parity summary is empty",),
            best_rank=None,
            best_row={},
            thresholds=_thresholds(config),
        )

    best = select_best_candidate(summary)
    reasons: list[str] = []

    if config.require_successful_best and _coerce_bool(best.get("failed", False)):
        error = _string_or_empty(best.get("error"))
        suffix = f": {error}" if error else ""
        reasons.append(f"best candidate is marked failed{suffix}")

    total_delta = _optional_float(best.get("count_abs_delta_total"))
    if total_delta is None:
        reasons.append("missing/non-finite count_abs_delta_total")
    elif total_delta > float(config.max_count_abs_delta_total):
        reasons.append(
            "count_abs_delta_total "
            f"{total_delta:g} exceeds {config.max_count_abs_delta_total:g}"
        )

    for column in config.required_count_delta_columns:
        delta = _optional_float(best.get(column))
        if delta is None:
            reasons.append(f"missing/non-finite required count delta column {column}")
        elif abs(delta) > float(config.max_stage_count_abs_delta):
            reasons.append(
                f"{column} abs delta {abs(delta):g} exceeds "
                f"{config.max_stage_count_abs_delta:g}"
            )

    if config.max_kf_mean_abs_delta_m is not None:
        kf_mean_delta = _optional_float(best.get("kf_all_steps_mean_abs_delta_m"))
        if kf_mean_delta is None:
            reasons.append("missing/non-finite kf_all_steps_mean_abs_delta_m")
        elif kf_mean_delta > float(config.max_kf_mean_abs_delta_m):
            reasons.append(
                "kf_all_steps_mean_abs_delta_m "
                f"{kf_mean_delta:g} exceeds {config.max_kf_mean_abs_delta_m:g}"
            )

    if config.max_paper_parity_score is not None:
        score = _optional_float(best.get("paper_parity_score"))
        if score is None:
            reasons.append("missing/non-finite paper_parity_score")
        elif score > float(config.max_paper_parity_score):
            reasons.append(
                f"paper_parity_score {score:g} exceeds {config.max_paper_parity_score:g}"
            )

    best_row = _row_to_mapping(best)
    return PaperParityGateResult(
        passed=not reasons,
        reasons=tuple(reasons),
        best_rank=_optional_int(best.get("rank")),
        best_row=best_row,
        thresholds=_thresholds(config),
    )


def select_best_candidate(summary: pd.DataFrame) -> pd.Series:
    """Return the row considered best by the grid summary ordering."""

    if summary.empty:
        raise ValueError("summary must not be empty")
    ranked = summary.copy()
    if "rank" in ranked.columns:
        rank = pd.to_numeric(ranked["rank"], errors="coerce")
        if rank.notna().any():
            ranked["_sort_rank"] = rank.fillna(np.inf)
            return ranked.sort_values("_sort_rank", kind="mergesort").iloc[0]

    if "failed" in ranked.columns:
        failed = ranked["failed"]
    else:
        failed = pd.Series(False, index=ranked.index)
    ranked["_sort_failed"] = failed.map(_coerce_bool)
    ranked["_sort_score"] = _numeric_sort_column(ranked, "paper_parity_score")
    ranked["_sort_count_delta"] = _numeric_sort_column(ranked, "count_abs_delta_total")
    ranked["_sort_kf_delta"] = _numeric_sort_column(ranked, "kf_all_steps_mean_abs_delta_m")
    return ranked.sort_values(
        ["_sort_failed", "_sort_score", "_sort_count_delta", "_sort_kf_delta"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).iloc[0]


def _numeric_sort_column(summary: pd.DataFrame, column: str) -> pd.Series:
    if column not in summary.columns:
        return pd.Series(np.inf, index=summary.index)
    return pd.to_numeric(summary[column], errors="coerce").fillna(np.inf)


def _thresholds(config: PaperParityGateConfig) -> dict[str, Any]:
    return {
        "max_count_abs_delta_total": int(config.max_count_abs_delta_total),
        "max_stage_count_abs_delta": int(config.max_stage_count_abs_delta),
        "max_kf_mean_abs_delta_m": config.max_kf_mean_abs_delta_m,
        "max_paper_parity_score": config.max_paper_parity_score,
        "require_successful_best": bool(config.require_successful_best),
        "required_count_delta_columns": list(config.required_count_delta_columns),
    }


def _parse_required_columns(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return DEFAULT_REQUIRED_COUNT_DELTA_COLUMNS
    columns: list[str] = []
    for raw_value in values:
        columns.extend(token.strip() for token in str(raw_value).split(",") if token.strip())
    return tuple(columns)


def _print_gate_result(result: PaperParityGateResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"{status}: paper-parity gate")
    if result.best_rank is not None:
        print(f"best_rank={result.best_rank}")
    best = result.best_row
    summary_parts = []
    for key in (
        "paper_parity_score",
        "count_abs_delta_total",
        "kf_all_steps_mean_abs_delta_m",
    ):
        if key in best and best[key] is not None:
            summary_parts.append(f"{key}={best[key]}")
    if summary_parts:
        print("best " + " ".join(summary_parts))
    if result.reasons:
        for reason in result.reasons:
            print(f"- {reason}")
    else:
        print("best candidate satisfies configured tolerances")


def _row_to_mapping(row: pd.Series) -> dict[str, Any]:
    return {
        str(key): _jsonable_value(value)
        for key, value in row.items()
        if not str(key).startswith("_sort_")
    }


def _jsonable_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable_value(value) for key, value in values.items()}


def _jsonable_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (np.ndarray, list, tuple)):
        return [_jsonable_value(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _optional_int(value: Any) -> int | None:
    scalar = _optional_float(value)
    return None if scalar is None else int(round(scalar))


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", ""}:
            return False
    return bool(value)


def _string_or_empty(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
