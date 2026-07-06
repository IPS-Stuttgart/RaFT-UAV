"""Prioritize MMUAD candidate-oracle failure blocks into action items.

The candidate-oracle block diagnostic separates temporal intervals into
``missing_good_candidate``, ``good_candidate_buried``, and ``covered_in_topk``.
This module turns those blocks into a compact experiment plan so the next run can
focus on the dominant failure mechanism rather than inspecting many block rows
manually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ACTION_BY_MODE = {
    "missing_good_candidate": "improve_extraction_or_calibration",
    "good_candidate_buried": "improve_topk_recall_or_ranker",
    "covered_in_topk": "improve_mixture_assignment_or_uncertainty",
}
_METHOD_BY_MODE = {
    "missing_good_candidate": (
        "compare raw/source-translated branches, relax pruning, add extraction/calibration branches"
    ),
    "good_candidate_buried": (
        "increase per-branch/per-source reservoir quotas, tune score offsets, retrain ranker recall"
    ),
    "covered_in_topk": (
        "tune learned sigma, Huber mixture weighting, assignment temperature, and responsibility floors"
    ),
}


def build_candidate_oracle_action_plan(
    blocks: pd.DataFrame,
    *,
    top_n_blocks: int = 20,
    duration_weight: float = 1.0,
    frame_weight: float = 1.0,
    error_weight: float = 1.0,
    rank_weight: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-block action rows and aggregated action summary.

    The priority score is intentionally simple and transparent.  It rewards long
    blocks, many frames, high oracle errors, and deeply buried oracle ranks.
    """

    rows = pd.DataFrame(blocks).copy()
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    _require_columns(rows, ["sequence_id", "oracle_failure_mode", "frame_count"])
    rows["frame_count"] = _numeric(rows, "frame_count", default=0.0)
    rows["duration_s"] = _numeric(rows, "duration_s", default=0.0)
    rows["oracle_all_3d_m_max"] = _numeric(rows, "oracle_all_3d_m_max", default=0.0)
    rows["oracle_all_rank_p95"] = _numeric(rows, "oracle_all_rank_p95", default=1.0)
    rows["recommended_action"] = rows["oracle_failure_mode"].map(_ACTION_BY_MODE).fillna(
        "inspect_block",
    )
    rows["recommended_method"] = rows["oracle_failure_mode"].map(_METHOD_BY_MODE).fillna(
        "inspect raw candidate rows and attribution diagnostics",
    )
    rows["action_priority_score"] = _priority_score(
        rows,
        duration_weight=duration_weight,
        frame_weight=frame_weight,
        error_weight=error_weight,
        rank_weight=rank_weight,
    )
    rows = rows.sort_values(
        ["action_priority_score", "frame_count", "duration_s"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    if top_n_blocks > 0:
        action_rows = rows.head(int(top_n_blocks)).copy()
    else:
        action_rows = rows.copy()
    action_rows["action_rank"] = np.arange(1, len(action_rows) + 1, dtype=int)
    summary = _action_summary(rows)
    return action_rows, summary


def write_candidate_oracle_action_plan_outputs(
    *,
    output_dir: Path,
    action_rows: pd.DataFrame,
    action_summary: pd.DataFrame,
) -> dict[str, str]:
    """Write action-plan artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "action_rows_csv": output_dir / "mmuad_candidate_oracle_action_plan.csv",
        "action_summary_csv": output_dir / "mmuad_candidate_oracle_action_summary.csv",
        "action_summary_json": output_dir / "mmuad_candidate_oracle_action_summary.json",
    }
    action_rows.to_csv(paths["action_rows_csv"], index=False)
    action_summary.to_csv(paths["action_summary_csv"], index=False)
    paths["action_summary_json"].write_text(
        json.dumps(
            {
                "actions": action_rows.to_dict(orient="records"),
                "summary": action_summary.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {key: str(value) for key, value in paths.items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-candidate-oracle-action-plan",
        description="prioritize MMUAD candidate-oracle failure blocks into experiment actions",
    )
    parser.add_argument("--blocks-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n-blocks", type=int, default=20)
    parser.add_argument("--duration-weight", type=float, default=1.0)
    parser.add_argument("--frame-weight", type=float, default=1.0)
    parser.add_argument("--error-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=0.25)
    args = parser.parse_args(argv)

    blocks = pd.read_csv(args.blocks_csv)
    action_rows, action_summary = build_candidate_oracle_action_plan(
        blocks,
        top_n_blocks=args.top_n_blocks,
        duration_weight=args.duration_weight,
        frame_weight=args.frame_weight,
        error_weight=args.error_weight,
        rank_weight=args.rank_weight,
    )
    paths = write_candidate_oracle_action_plan_outputs(
        output_dir=args.output_dir,
        action_rows=action_rows,
        action_summary=action_summary,
    )
    print("mmuad_candidate_oracle_action_plan=ok")
    print(f"action_rows={len(action_rows)}")
    print(f"action_summary_rows={len(action_summary)}")
    if not action_summary.empty:
        first = action_summary.iloc[0]
        print(f"top_recommended_action={first['recommended_action']}")
    for key, value in paths.items():
        print(f"{key}={value}")
    return 0


def _priority_score(
    rows: pd.DataFrame,
    *,
    duration_weight: float,
    frame_weight: float,
    error_weight: float,
    rank_weight: float,
) -> pd.Series:
    duration = _normalize(rows["duration_s"])
    frames = _normalize(rows["frame_count"])
    error = _normalize(rows["oracle_all_3d_m_max"])
    rank = _normalize(rows["oracle_all_rank_p95"])
    return (
        float(duration_weight) * duration
        + float(frame_weight) * frames
        + float(error_weight) * error
        + float(rank_weight) * rank
    )


def _action_summary(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for action, group in rows.groupby("recommended_action", sort=False):
        records.append(
            {
                "recommended_action": str(action),
                "recommended_method": str(group["recommended_method"].iloc[0]),
                "block_count": int(len(group)),
                "frame_count": int(pd.to_numeric(group["frame_count"], errors="coerce").sum()),
                "duration_s_sum": float(pd.to_numeric(group["duration_s"], errors="coerce").sum()),
                "max_block_error_m": _max(group["oracle_all_3d_m_max"]),
                "p95_oracle_rank_p95": _quantile(group["oracle_all_rank_p95"], 0.95),
                "priority_score_sum": float(
                    pd.to_numeric(group["action_priority_score"], errors="coerce").sum(),
                ),
            },
        )
    summary = pd.DataFrame.from_records(records)
    return summary.sort_values(
        ["priority_score_sum", "frame_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _require_columns(rows: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise ValueError(f"oracle block rows missing required columns: {missing}")


def _numeric(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce").fillna(default).astype(float)
    return pd.Series(default, index=rows.index, dtype=float)


def _normalize(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)
    max_value = float(values.max()) if len(values) else 0.0
    if not np.isfinite(max_value) or max_value <= 0.0:
        return pd.Series(0.0, index=values.index, dtype=float)
    return values / max_value


def _max(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.max()) if not numeric.empty else float("nan")


def _quantile(values: pd.Series, quantile: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.quantile(quantile)) if not numeric.empty else float("nan")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
