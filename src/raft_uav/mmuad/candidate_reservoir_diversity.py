"""Diversity-preserving post-cap for MMUAD candidate reservoirs.

The standard reservoir builder keeps global top-N, per-source, and per-branch
candidates before applying a final per-frame cap. A strict score-only final cap
can still discard the low-score raw/dynamic/calibrated branch that was explicitly
kept to preserve oracle recall. This helper post-processes any reservoir CSV so
that the final cap keeps a small quota per source and per branch before filling
the remaining slots by score.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from raft_uav.mmuad.candidate_reservoir import build_oracle_recall_tables
from raft_uav.mmuad.schema import normalize_candidate_columns, normalize_truth_columns


def diversity_cap_reservoir(
    reservoir: pd.DataFrame,
    *,
    max_candidates_per_frame: int = 40,
    min_per_source: int = 1,
    min_per_branch: int = 1,
    score_column: str = "candidate_reservoir_score",
    fallback_score_column: str = "confidence",
    branch_column: str = "candidate_branch",
) -> pd.DataFrame:
    """Return a reservoir capped per frame while preserving branch/source diversity.

    The function first reserves the best rows per source and candidate branch,
    then fills the remaining budget by score. If the protected set itself is
    larger than the frame budget, a rarity-aware quota selector is used instead
    of a score-only truncation. This prevents a common high-score source/branch
    from crowding out a rare branch that was deliberately retained upstream.
    """

    rows = normalize_candidate_columns(pd.DataFrame(reservoir).copy())
    if rows.empty:
        return rows.assign(
            candidate_diversity_cap_reason=pd.Series(dtype=str),
            candidate_diversity_cap_rank=pd.Series(dtype=float),
        )
    rows = rows.copy().reset_index(drop=True)
    if branch_column not in rows.columns:
        rows[branch_column] = rows.get("candidate_branch", rows.get("source", "candidate"))
    rows[branch_column] = rows[branch_column].fillna("candidate").astype(str)
    if "source" not in rows.columns:
        rows["source"] = "candidate"
    rows["source"] = rows["source"].fillna("candidate").astype(str)
    rows["_diversity_row_id"] = np.arange(len(rows), dtype=int)
    rows["_diversity_stable_key"] = _stable_candidate_key(rows, branch_column=branch_column)
    rows["_diversity_score"] = _score(rows, score_column, fallback_score_column)

    parts: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=False):
        protected: set[int] = set()
        reasons: dict[int, set[str]] = {}
        if int(min_per_source) > 0:
            _protect_group_topn(
                frame,
                group_column="source",
                count=int(min_per_source),
                reason_prefix="source",
                protected=protected,
                reasons=reasons,
            )
        if int(min_per_branch) > 0:
            _protect_group_topn(
                frame,
                group_column=branch_column,
                count=int(min_per_branch),
                reason_prefix="branch",
                protected=protected,
                reasons=reasons,
            )

        ranked_frame = frame.sort_values(
            ["_diversity_score", "_diversity_stable_key", "_diversity_row_id"],
            ascending=[False, True, True],
        )
        if int(max_candidates_per_frame) <= 0:
            selected = ranked_frame.copy()
            for row_id in selected["_diversity_row_id"].astype(int):
                reasons.setdefault(int(row_id), set()).add("score_unbounded")
        else:
            budget = int(max_candidates_per_frame)
            protected_rows = frame.loc[frame["_diversity_row_id"].isin(protected)].copy()
            if len(protected_rows) >= budget:
                selected = _select_protected_quota_rows(
                    protected_rows,
                    budget=budget,
                    branch_column=branch_column,
                    min_per_source=max(int(min_per_source), 0),
                    min_per_branch=max(int(min_per_branch), 0),
                )
                for row_id in selected["_diversity_row_id"].astype(int):
                    reasons.setdefault(int(row_id), set()).add("protected_quota_cap")
            else:
                selected_ids = set(protected_rows["_diversity_row_id"].astype(int))
                fill = ranked_frame.loc[~ranked_frame["_diversity_row_id"].isin(selected_ids)].head(
                    max(budget - len(selected_ids), 0),
                )
                for row_id in fill["_diversity_row_id"].astype(int):
                    reasons.setdefault(int(row_id), set()).add("score_fill")
                selected = pd.concat([protected_rows, fill], ignore_index=False)
        selected = selected.sort_values(
            ["_diversity_score", "_diversity_stable_key", "_diversity_row_id"],
            ascending=[False, True, True],
        ).copy()
        selected["candidate_diversity_cap_rank"] = np.arange(1, len(selected) + 1, dtype=float)
        selected["candidate_diversity_cap_reason"] = [
            ";".join(sorted(reasons.get(int(row_id), {"score_fill"})))
            for row_id in selected["_diversity_row_id"].astype(int)
        ]
        parts.append(selected)

    out = pd.concat(parts, ignore_index=True) if parts else rows.iloc[0:0].copy()
    out = out.drop(
        columns=["_diversity_row_id", "_diversity_stable_key", "_diversity_score"],
        errors="ignore",
    )
    return out.sort_values(["sequence_id", "time_s", "candidate_diversity_cap_rank"]).reset_index(
        drop=True,
    )


def _select_protected_quota_rows(
    protected_rows: pd.DataFrame,
    *,
    budget: int,
    branch_column: str,
    min_per_source: int,
    min_per_branch: int,
) -> pd.DataFrame:
    """Select protected rows while satisfying rare source/branch quotas first.

    When protected rows exceed the hard cap, a plain score sort can erase an
    entire low-score source or branch. The selector repeatedly chooses the
    scarcest still-unmet quota and then the candidate that covers the most unmet
    quotas, with score used only as a tie-breaker. Remaining slots are filled by
    score. The algorithm is deterministic and truth-free.
    """

    if budget <= 0 or protected_rows.empty:
        return protected_rows.iloc[0:0].copy()
    rows = protected_rows.copy()
    rows["source"] = rows["source"].fillna("candidate").astype(str)
    rows[branch_column] = rows[branch_column].fillna("candidate").astype(str)
    rows = rows.sort_values(
        ["_diversity_score", "_diversity_stable_key", "_diversity_row_id"],
        ascending=[False, True, True],
    )

    source_targets = {
        value: min(int(min_per_source), int(count))
        for value, count in rows["source"].value_counts(dropna=False).items()
        if min_per_source > 0
    }
    branch_targets = {
        value: min(int(min_per_branch), int(count))
        for value, count in rows[branch_column].value_counts(dropna=False).items()
        if min_per_branch > 0
    }
    source_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    selected_ids: list[int] = []
    available_ids = set(rows["_diversity_row_id"].astype(int))
    exhausted: set[tuple[str, str]] = set()

    while len(selected_ids) < budget and available_ids:
        unmet = _unmet_quota_labels(
            source_targets,
            branch_targets,
            source_counts,
            branch_counts,
            exhausted=exhausted,
        )
        if not unmet:
            break
        label_candidates: list[tuple[int, str, str, pd.DataFrame]] = []
        remaining = rows.loc[rows["_diversity_row_id"].isin(available_ids)]
        for kind, value in unmet:
            column = "source" if kind == "source" else branch_column
            eligible = remaining.loc[remaining[column] == value]
            if eligible.empty:
                exhausted.add((kind, value))
                continue
            label_candidates.append((len(eligible), kind, value, eligible))
        if not label_candidates:
            break
        _, _, _, eligible = min(
            label_candidates,
            key=lambda item: (item[0], item[1], item[2]),
        )
        ranked = eligible.copy()
        ranked["_quota_coverage_gain"] = [
            _quota_coverage_gain(
                row,
                branch_column=branch_column,
                source_targets=source_targets,
                branch_targets=branch_targets,
                source_counts=source_counts,
                branch_counts=branch_counts,
            )
            for _, row in ranked.iterrows()
        ]
        chosen = ranked.sort_values(
            [
                "_quota_coverage_gain",
                "_diversity_score",
                "_diversity_stable_key",
                "_diversity_row_id",
            ],
            ascending=[False, False, True, True],
        ).iloc[0]
        row_id = int(chosen["_diversity_row_id"])
        selected_ids.append(row_id)
        available_ids.remove(row_id)
        source_counts[str(chosen["source"])] += 1
        branch_counts[str(chosen[branch_column])] += 1

    if len(selected_ids) < budget and available_ids:
        fill = rows.loc[rows["_diversity_row_id"].isin(available_ids)].head(
            budget - len(selected_ids),
        )
        selected_ids.extend(fill["_diversity_row_id"].astype(int).tolist())

    selected_order = {row_id: index for index, row_id in enumerate(selected_ids)}
    selected = rows.loc[rows["_diversity_row_id"].isin(selected_ids)].copy()
    selected["_quota_selection_order"] = selected["_diversity_row_id"].map(selected_order)
    return selected.sort_values("_quota_selection_order").drop(
        columns=["_quota_selection_order"],
        errors="ignore",
    )


def _unmet_quota_labels(
    source_targets: dict[str, int],
    branch_targets: dict[str, int],
    source_counts: Counter[str],
    branch_counts: Counter[str],
    *,
    exhausted: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    labels.extend(
        ("source", value)
        for value, target in source_targets.items()
        if source_counts[value] < target and ("source", value) not in exhausted
    )
    labels.extend(
        ("branch", value)
        for value, target in branch_targets.items()
        if branch_counts[value] < target and ("branch", value) not in exhausted
    )
    return labels


def _quota_coverage_gain(
    row: pd.Series,
    *,
    branch_column: str,
    source_targets: dict[str, int],
    branch_targets: dict[str, int],
    source_counts: Counter[str],
    branch_counts: Counter[str],
) -> int:
    source = str(row["source"])
    branch = str(row[branch_column])
    return int(source_counts[source] < source_targets.get(source, 0)) + int(
        branch_counts[branch] < branch_targets.get(branch, 0)
    )


def diversity_cap_summary(input_rows: pd.DataFrame, output_rows: pd.DataFrame) -> dict[str, Any]:
    """Build a compact summary for a diversity-capped reservoir."""

    branch_coverage, source_coverage = _frame_label_coverage(input_rows, output_rows)
    return {
        "input_rows": int(len(input_rows)),
        "output_rows": int(len(output_rows)),
        "input_frame_count": int(_frame_counts(input_rows).size),
        "output_frame_count": int(_frame_counts(output_rows).size),
        "input_candidates_per_frame_mean": _mean(_frame_counts(input_rows)),
        "output_candidates_per_frame_mean": _mean(_frame_counts(output_rows)),
        "output_candidates_per_frame_p95": _quantile(_frame_counts(output_rows), 0.95),
        "output_candidates_per_frame_max": _max(_frame_counts(output_rows)),
        "mean_branch_coverage_fraction": _mean(pd.Series(branch_coverage, dtype=float)),
        "mean_source_coverage_fraction": _mean(pd.Series(source_coverage, dtype=float)),
        "frames_all_branches_preserved_fraction": _all_preserved_fraction(branch_coverage),
        "frames_all_sources_preserved_fraction": _all_preserved_fraction(source_coverage),
        "source_counts": _value_counts(output_rows, "source"),
        "branch_counts": _value_counts(output_rows, "candidate_branch"),
        "diversity_cap_reason_counts": _reason_counts(output_rows),
    }


def write_diversity_cap_outputs(
    capped: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    input_rows: pd.DataFrame | None = None,
) -> None:
    """Write capped reservoir outputs."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    capped.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(
                diversity_cap_summary(input_rows if input_rows is not None else capped, capped),
                indent=2,
            ),
            encoding="utf-8",
        )


def _protect_group_topn(
    frame: pd.DataFrame,
    *,
    group_column: str,
    count: int,
    reason_prefix: str,
    protected: set[int],
    reasons: dict[int, set[str]],
) -> None:
    if group_column not in frame.columns or count <= 0:
        return
    for value, group in frame.groupby(group_column, sort=False, dropna=False):
        selected = group.sort_values(
            ["_diversity_score", "_diversity_stable_key", "_diversity_row_id"],
            ascending=[False, True, True],
        ).head(int(count))
        for row_id in selected["_diversity_row_id"].astype(int):
            protected.add(int(row_id))
            reasons.setdefault(int(row_id), set()).add(f"{reason_prefix}:{value}")


def _frame_label_coverage(
    input_rows: pd.DataFrame,
    output_rows: pd.DataFrame,
) -> tuple[list[float], list[float]]:
    input_frame_map = {
        (str(sequence_id), float(time_s)): group
        for (sequence_id, time_s), group in pd.DataFrame(input_rows).groupby(
            ["sequence_id", "time_s"],
            sort=False,
        )
    }
    output_frame_map = {
        (str(sequence_id), float(time_s)): group
        for (sequence_id, time_s), group in pd.DataFrame(output_rows).groupby(
            ["sequence_id", "time_s"],
            sort=False,
        )
    }
    branch_coverage: list[float] = []
    source_coverage: list[float] = []
    for key, frame in input_frame_map.items():
        output = output_frame_map.get(key, pd.DataFrame())
        branch_coverage.append(_label_coverage_fraction(frame, output, "candidate_branch"))
        source_coverage.append(_label_coverage_fraction(frame, output, "source"))
    return branch_coverage, source_coverage


def _label_coverage_fraction(
    input_rows: pd.DataFrame,
    output_rows: pd.DataFrame,
    column: str,
) -> float:
    if column not in input_rows.columns:
        return 1.0
    labels = set(input_rows[column].fillna("candidate").astype(str))
    if not labels:
        return 1.0
    selected = (
        set(output_rows[column].fillna("candidate").astype(str))
        if not output_rows.empty and column in output_rows.columns
        else set()
    )
    return float(len(labels & selected) / len(labels))


def _all_preserved_fraction(coverage: list[float]) -> float:
    if not coverage:
        return 0.0
    return float(np.mean(np.asarray(coverage, dtype=float) >= 1.0))


def _stable_candidate_key(rows: pd.DataFrame, *, branch_column: str) -> pd.Series:
    """Return an input-order-independent candidate key for deterministic ties."""

    track = rows.get("track_id", pd.Series("", index=rows.index)).fillna("").astype(str)
    source = (
        rows.get("source", pd.Series("candidate", index=rows.index))
        .fillna("candidate")
        .astype(str)
    )
    branch = rows.get(
        branch_column,
        pd.Series("candidate", index=rows.index),
    ).fillna("candidate").astype(str)
    coordinate_parts = []
    for column in ("x_m", "y_m", "z_m"):
        numeric = pd.to_numeric(
            rows.get(column, pd.Series(np.nan, index=rows.index)),
            errors="coerce",
        )
        coordinate_parts.append(
            numeric.map(
                lambda value: f"{float(value):.12g}" if np.isfinite(value) else "nan"
            )
        )
    return (
        branch
        + "|"
        + source
        + "|"
        + track
        + "|"
        + coordinate_parts[0]
        + "|"
        + coordinate_parts[1]
        + "|"
        + coordinate_parts[2]
    )


def _score(rows: pd.DataFrame, score_column: str, fallback_score_column: str) -> pd.Series:
    primary = _numeric(rows, score_column, default=np.nan)
    fallback = _numeric(rows, fallback_score_column, default=1.0)
    return primary.fillna(fallback).fillna(0.0).astype(float)


def _numeric(rows: pd.DataFrame, column: str, *, default: float) -> pd.Series:
    if column in rows.columns:
        return pd.to_numeric(rows[column], errors="coerce")
    return pd.Series(default, index=rows.index, dtype=float)


def _frame_counts(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=int)
    return rows.groupby(["sequence_id", "time_s"], dropna=False).size()


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def _quantile(values: pd.Series, q: float) -> float:
    return float(values.quantile(q)) if not values.empty else 0.0


def _max(values: pd.Series) -> int:
    return int(values.max()) if not values.empty else 0


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in rows.columns:
        return {}
    return {str(key): int(value) for key, value in rows[column].value_counts(dropna=False).items()}


def _reason_counts(rows: pd.DataFrame) -> dict[str, int]:
    column = "candidate_diversity_cap_reason"
    if column not in rows.columns:
        return {}
    counts: dict[str, int] = {}
    for value in rows[column].dropna().astype(str):
        for reason in value.replace(",", ";").split(";"):
            reason = reason.strip()
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="raft-uav-mmuad-diversity-cap-reservoir",
        description="apply a diversity-preserving final cap to an MMUAD candidate reservoir",
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--truth-csv", type=Path)
    parser.add_argument("--oracle-frame-csv", type=Path)
    parser.add_argument("--oracle-summary-csv", type=Path)
    parser.add_argument("--oracle-by-sequence-csv", type=Path)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--min-per-source", type=int, default=1)
    parser.add_argument("--min-per-branch", type=int, default=1)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--fallback-score-column", default="confidence")
    parser.add_argument("--branch-column", default="candidate_branch")
    parser.add_argument("--top-k", type=int, action="append", default=[1, 3, 5, 10, 20])
    parser.add_argument("--max-truth-time-delta-s", type=float, default=0.5)
    args = parser.parse_args(argv)

    rows = pd.read_csv(args.input_csv)
    capped = diversity_cap_reservoir(
        rows,
        max_candidates_per_frame=args.max_candidates_per_frame,
        min_per_source=args.min_per_source,
        min_per_branch=args.min_per_branch,
        score_column=args.score_column,
        fallback_score_column=args.fallback_score_column,
        branch_column=args.branch_column,
    )
    write_diversity_cap_outputs(
        capped,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        input_rows=rows,
    )
    print("mmuad_diversity_cap_reservoir=ok")
    print(f"input_rows={len(rows)}")
    print(f"output_rows={len(capped)}")
    print(f"output_csv={args.output_csv}")

    if args.truth_csv is not None:
        truth = normalize_truth_columns(pd.read_csv(args.truth_csv))
        frame_rows, pooled, by_sequence = build_oracle_recall_tables(
            capped,
            truth,
            top_k_values=tuple(args.top_k),
            max_truth_time_delta_s=args.max_truth_time_delta_s,
        )
        if args.oracle_frame_csv is not None:
            args.oracle_frame_csv.parent.mkdir(parents=True, exist_ok=True)
            frame_rows.to_csv(args.oracle_frame_csv, index=False)
        if args.oracle_summary_csv is not None:
            args.oracle_summary_csv.parent.mkdir(parents=True, exist_ok=True)
            pooled.to_csv(args.oracle_summary_csv, index=False)
        if args.oracle_by_sequence_csv is not None:
            args.oracle_by_sequence_csv.parent.mkdir(parents=True, exist_ok=True)
            by_sequence.to_csv(args.oracle_by_sequence_csv, index=False)
        if not pooled.empty:
            print(f"oracle_all_mse={pooled.loc[0, 'oracle_all_3d_m_mse']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
