"""Classify reservoir-mixture bottlenecks for MMUAD tuning loops.

Reservoir-mixture runs now expose two complementary signals: the achieved
mixture trajectory error and the retained-reservoir oracle ceiling.  This module
turns those gap tables into an explicit action-oriented bottleneck label so the
next experiment can distinguish:

* retained candidate pool is not good enough;
* good candidates survived, but the mixture assignment did not use them;
* small/top-K reservoir quotas are hiding candidates that exist deeper in the
  retained pool.

The utility is diagnostic only. It consumes already-written summary CSVs and
never uses truth unless the upstream diagnostic CSV was generated with truth.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BOTTLENECK_UNKNOWN = "unknown"
BOTTLENECK_RESERVOIR_CEILING = "reservoir_ceiling_limited"
BOTTLENECK_ASSIGNMENT = "assignment_limited"
BOTTLENECK_TOPK_RECALL = "topk_recall_limited"
BOTTLENECK_MIXED = "mixed"
BOTTLENECK_NEAR_ORACLE = "near_oracle"

ACTION_BY_BOTTLENECK = {
    BOTTLENECK_UNKNOWN: "inspect_missing_metrics",
    BOTTLENECK_RESERVOIR_CEILING: "add_or_repair_candidate_branches",
    BOTTLENECK_ASSIGNMENT: "improve_mixture_weighting_sigma_or_assignment",
    BOTTLENECK_TOPK_RECALL: "increase_reservoir_quota_or_rebalance_branch_scores",
    BOTTLENECK_MIXED: "jointly_improve_reservoir_and_assignment",
    BOTTLENECK_NEAR_ORACLE: "freeze_or_shift_to_candidate_generation",
}


@dataclass(frozen=True)
class BottleneckConfig:
    """Thresholds for reservoir-mixture bottleneck classification."""

    near_oracle_ratio: float = 1.15
    assignment_ratio: float = 2.0
    assignment_fraction: float = 0.50
    topk_recall_ratio: float = 1.50
    topk_recall_absolute_gap_mse: float = 10.0
    target_mse_3d_m2: float | None = None


def classify_gap_row(row: pd.Series | dict[str, Any], *, config: BottleneckConfig | None = None) -> dict[str, Any]:
    """Classify one reservoir-mixture gap row and return diagnostic fields."""

    config = config or BottleneckConfig()
    data = dict(row)
    mixture_mse = _optional_float(data.get("mixture_mse_3d_m2"))
    oracle_all_mse = _optional_float(data.get("reservoir_oracle_all_mse_3d_m2"))
    best_topk_mse = _optional_float(data.get("best_reservoir_oracle_topk_mse_3d_m2"))
    if best_topk_mse is None:
        best_topk_mse = _best_topk_mse(data)

    result: dict[str, Any] = {
        "mixture_mse_3d_m2": mixture_mse,
        "reservoir_oracle_all_mse_3d_m2": oracle_all_mse,
        "best_reservoir_oracle_topk_mse_3d_m2": best_topk_mse,
        "target_mse_3d_m2": config.target_mse_3d_m2,
    }
    if mixture_mse is None or oracle_all_mse is None:
        result.update(
            {
                "primary_bottleneck": BOTTLENECK_UNKNOWN,
                "recommended_action": ACTION_BY_BOTTLENECK[BOTTLENECK_UNKNOWN],
            }
        )
        return _jsonable(result)

    assignment_gap = mixture_mse - oracle_all_mse
    assignment_fraction = _safe_ratio(assignment_gap, mixture_mse)
    mixture_to_all_ratio = _safe_ratio(mixture_mse, oracle_all_mse)
    target_ratio = (
        _safe_ratio(oracle_all_mse, config.target_mse_3d_m2)
        if config.target_mse_3d_m2 is not None
        else None
    )
    topk_gap = None
    topk_ratio = None
    if best_topk_mse is not None:
        topk_gap = best_topk_mse - oracle_all_mse
        topk_ratio = _safe_ratio(best_topk_mse, oracle_all_mse)

    bottleneck = _classify_bottleneck(
        mixture_mse=mixture_mse,
        oracle_all_mse=oracle_all_mse,
        assignment_gap=assignment_gap,
        assignment_fraction=assignment_fraction,
        mixture_to_all_ratio=mixture_to_all_ratio,
        best_topk_mse=best_topk_mse,
        topk_gap=topk_gap,
        topk_ratio=topk_ratio,
        config=config,
    )
    result.update(
        {
            "primary_bottleneck": bottleneck,
            "recommended_action": ACTION_BY_BOTTLENECK[bottleneck],
            "assignment_gap_mse_3d_m2": assignment_gap,
            "assignment_gap_fraction_of_mixture_mse": assignment_fraction,
            "mixture_to_reservoir_oracle_all_ratio": mixture_to_all_ratio,
            "topk_recall_gap_mse_3d_m2": topk_gap,
            "topk_recall_ratio": topk_ratio,
            "reservoir_oracle_all_to_target_ratio": target_ratio,
            "reservoir_oracle_all_beats_target": (
                None
                if config.target_mse_3d_m2 is None
                else bool(oracle_all_mse <= float(config.target_mse_3d_m2))
            ),
        }
    )
    return _jsonable(result)


def annotate_gap_table(gap_rows: pd.DataFrame, *, config: BottleneckConfig | None = None) -> pd.DataFrame:
    """Append bottleneck classification columns to a gap summary table."""

    rows = pd.DataFrame(gap_rows).copy()
    if rows.empty:
        return rows.assign(
            primary_bottleneck=pd.Series(dtype=str),
            recommended_action=pd.Series(dtype=str),
        )
    annotations = pd.DataFrame.from_records(
        [classify_gap_row(record, config=config) for record in rows.to_dict(orient="records")]
    )
    add_columns = [
        column for column in annotations.columns if column not in rows.columns or column in _OVERWRITE_COLUMNS
    ]
    for column in add_columns:
        rows[column] = annotations[column]
    return rows


def build_bottleneck_summary(annotated: pd.DataFrame, *, config: BottleneckConfig) -> dict[str, Any]:
    """Return a compact JSON summary for annotated gap rows."""

    rows = pd.DataFrame(annotated).copy()
    if rows.empty:
        return {"row_count": 0, "config": asdict(config), "bottleneck_counts": {}}
    return _jsonable(
        {
            "row_count": int(len(rows)),
            "config": asdict(config),
            "bottleneck_counts": _value_counts(rows, "primary_bottleneck"),
            "action_counts": _value_counts(rows, "recommended_action"),
            "worst_assignment_gap": _max_record(rows, "assignment_gap_mse_3d_m2"),
            "worst_topk_recall_gap": _max_record(rows, "topk_recall_gap_mse_3d_m2"),
            "worst_reservoir_ceiling": _max_record(
                rows,
                "reservoir_oracle_all_mse_3d_m2",
            ),
        }
    )


def write_bottleneck_outputs(
    annotated: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    config: BottleneckConfig | None = None,
) -> dict[str, Path]:
    """Write annotated bottleneck rows and optional JSON summary."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(output_csv, index=False)
    paths = {"output_csv": output_csv}
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = build_bottleneck_summary(annotated, config=config or BottleneckConfig())
        summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        paths["summary_json"] = summary_json
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-reservoir-bottleneck-audit",
        description="classify MMUAD reservoir-mixture gaps into action-oriented bottlenecks",
    )
    parser.add_argument("--gap-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--target-mse-3d-m2", type=float)
    parser.add_argument("--near-oracle-ratio", type=float, default=1.15)
    parser.add_argument("--assignment-ratio", type=float, default=2.0)
    parser.add_argument("--assignment-fraction", type=float, default=0.50)
    parser.add_argument("--topk-recall-ratio", type=float, default=1.50)
    parser.add_argument("--topk-recall-absolute-gap-mse", type=float, default=10.0)
    args = parser.parse_args(argv)

    config = BottleneckConfig(
        near_oracle_ratio=float(args.near_oracle_ratio),
        assignment_ratio=float(args.assignment_ratio),
        assignment_fraction=float(args.assignment_fraction),
        topk_recall_ratio=float(args.topk_recall_ratio),
        topk_recall_absolute_gap_mse=float(args.topk_recall_absolute_gap_mse),
        target_mse_3d_m2=args.target_mse_3d_m2,
    )
    annotated = annotate_gap_table(pd.read_csv(args.gap_csv), config=config)
    paths = write_bottleneck_outputs(
        annotated,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        config=config,
    )
    print("mmuad_reservoir_bottleneck_audit=ok")
    print(f"row_count={len(annotated)}")
    counts = _value_counts(annotated, "primary_bottleneck")
    for key, value in sorted(counts.items()):
        print(f"{key}={value}")
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


_OVERWRITE_COLUMNS = {
    "mixture_mse_3d_m2",
    "reservoir_oracle_all_mse_3d_m2",
    "best_reservoir_oracle_topk_mse_3d_m2",
}


def _classify_bottleneck(
    *,
    mixture_mse: float,
    oracle_all_mse: float,
    assignment_gap: float,
    assignment_fraction: float | None,
    mixture_to_all_ratio: float | None,
    best_topk_mse: float | None,
    topk_gap: float | None,
    topk_ratio: float | None,
    config: BottleneckConfig,
) -> str:
    target = config.target_mse_3d_m2
    if mixture_mse <= 1.0e-12 and oracle_all_mse <= 1.0e-12:
        return BOTTLENECK_NEAR_ORACLE
    if oracle_all_mse <= 1.0e-12:
        return BOTTLENECK_ASSIGNMENT
    if _topk_recall_is_bottleneck(
        oracle_all_mse=oracle_all_mse,
        best_topk_mse=best_topk_mse,
        topk_gap=topk_gap,
        topk_ratio=topk_ratio,
        config=config,
    ):
        return BOTTLENECK_TOPK_RECALL
    if assignment_gap <= 0.0:
        return BOTTLENECK_RESERVOIR_CEILING
    if mixture_to_all_ratio is not None and mixture_to_all_ratio <= float(config.near_oracle_ratio):
        return BOTTLENECK_RESERVOIR_CEILING
    if target is not None and oracle_all_mse > float(target) and (assignment_fraction or 0.0) < float(
        config.assignment_fraction
    ):
        return BOTTLENECK_RESERVOIR_CEILING
    if (
        mixture_to_all_ratio is not None
        and mixture_to_all_ratio >= float(config.assignment_ratio)
        and (assignment_fraction or 0.0) >= float(config.assignment_fraction)
    ):
        return BOTTLENECK_ASSIGNMENT
    return BOTTLENECK_MIXED


def _topk_recall_is_bottleneck(
    *,
    oracle_all_mse: float,
    best_topk_mse: float | None,
    topk_gap: float | None,
    topk_ratio: float | None,
    config: BottleneckConfig,
) -> bool:
    if best_topk_mse is None or topk_gap is None:
        return False
    target = config.target_mse_3d_m2
    if target is not None and oracle_all_mse <= float(target) < best_topk_mse:
        return True
    return bool(
        topk_gap >= float(config.topk_recall_absolute_gap_mse)
        and topk_ratio is not None
        and topk_ratio >= float(config.topk_recall_ratio)
    )


def _best_topk_mse(data: dict[str, Any]) -> float | None:
    candidates: list[float] = []
    for key, value in data.items():
        key_text = str(key)
        if not key_text.startswith("reservoir_oracle_top"):
            continue
        if not key_text.endswith("_mse_3d_m2"):
            continue
        number = _optional_float(value)
        if number is not None:
            candidates.append(number)
    return min(candidates) if candidates else None


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(number) else number


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator) / float(denominator)


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {str(key): int(value) for key, value in rows[column].value_counts(dropna=False).items()}


def _max_record(rows: pd.DataFrame, column: str) -> dict[str, Any]:
    if rows.empty or column not in rows.columns:
        return {}
    values = pd.to_numeric(rows[column], errors="coerce")
    if values.dropna().empty:
        return {}
    return _jsonable(rows.loc[int(values.idxmax())].to_dict())


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
