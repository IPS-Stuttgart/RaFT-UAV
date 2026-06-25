"""Build branch-preserving MMUAD candidate reservoirs.

This utility keeps multiple candidate-generation branches alive for downstream
mixture-MAP or oracle-ceiling experiments.  It is intentionally light-weight:
rows are loaded from one or more candidate CSVs, annotated with a
``candidate_branch`` column, and then selected per ``(sequence_id, time_s)`` by a
union of per-source, per-branch, and global top-scoring rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

_REQUIRED_COLUMNS = ("sequence_id", "time_s", "x_m", "y_m", "z_m")
_DEFAULT_SCORE_COLUMNS = ("ranker_score", "confidence", "raw_confidence")


def load_candidate_inputs(specs: Sequence[str]) -> pd.DataFrame:
    """Load candidate CSV specs and attach branch metadata.

    Specs may be either ``path/to/candidates.csv`` or
    ``branch_name=path/to/candidates.csv``.  Existing ``candidate_branch`` values
    are preserved, with missing values filled from the spec branch.
    """

    frames: list[pd.DataFrame] = []
    for spec in specs:
        branch, path = _split_candidate_spec(spec)
        rows = pd.read_csv(path)
        if rows.empty:
            continue
        _validate_required_columns(rows, path)
        rows = rows.copy()
        if "source" not in rows.columns:
            rows["source"] = "unknown"
        if "track_id" not in rows.columns:
            rows["track_id"] = np.arange(len(rows), dtype=int).astype(str)
        if "candidate_branch" not in rows.columns:
            rows["candidate_branch"] = branch
        else:
            rows["candidate_branch"] = rows["candidate_branch"].fillna(branch).astype(str)
            rows.loc[rows["candidate_branch"].str.len() == 0, "candidate_branch"] = branch
        if "original_x_m" not in rows.columns:
            rows["original_x_m"] = pd.to_numeric(rows["x_m"], errors="coerce")
            rows["original_y_m"] = pd.to_numeric(rows["y_m"], errors="coerce")
            rows["original_z_m"] = pd.to_numeric(rows["z_m"], errors="coerce")
        rows["candidate_branch_input_path"] = str(path)
        frames.append(rows)
    if not frames:
        return pd.DataFrame(columns=[*_REQUIRED_COLUMNS, "source", "candidate_branch"])
    return pd.concat(frames, ignore_index=True)


def build_candidate_reservoir(
    candidates: pd.DataFrame,
    *,
    top_per_source: int = 3,
    top_per_branch: int = 3,
    global_top_n: int = 40,
    max_candidates_per_frame: int | None = None,
    score_columns: Sequence[str] = _DEFAULT_SCORE_COLUMNS,
    score_floor_quantile: float | None = None,
) -> pd.DataFrame:
    """Return a branch-preserving reservoir of candidates.

    The reservoir is the union of:

    * top ``top_per_source`` rows per source,
    * top ``top_per_branch`` rows per candidate branch,
    * top ``global_top_n`` rows globally within the frame,
    * optional score-floor rows above ``score_floor_quantile``.
    """

    if candidates.empty:
        return candidates.copy()
    rows = candidates.copy().reset_index(drop=True)
    _validate_required_columns(rows, Path("<dataframe>"))
    if "source" not in rows.columns:
        rows["source"] = "unknown"
    if "candidate_branch" not in rows.columns:
        rows["candidate_branch"] = "default"
    rows["_candidate_reservoir_input_index"] = np.arange(len(rows), dtype=int)
    rows["candidate_reservoir_score"] = _candidate_scores(rows, score_columns)

    selected_frames: list[pd.DataFrame] = []
    for _, frame in rows.groupby(["sequence_id", "time_s"], sort=True, dropna=False):
        reasons: dict[int, set[str]] = {}
        frame = frame.copy()
        eligible = frame
        if score_floor_quantile is not None:
            quantile = min(max(float(score_floor_quantile), 0.0), 1.0)
            threshold = float(frame["candidate_reservoir_score"].quantile(quantile))
            floor_rows = frame.loc[frame["candidate_reservoir_score"] >= threshold]
            eligible = floor_rows
            _add_rows(reasons, floor_rows, "score_floor")
        if top_per_source > 0:
            for source, group in eligible.groupby("source", sort=True, dropna=False):
                _add_top_rows(reasons, group, int(top_per_source), f"source:{source}")
        if top_per_branch > 0:
            for branch, group in eligible.groupby("candidate_branch", sort=True, dropna=False):
                _add_top_rows(reasons, group, int(top_per_branch), f"branch:{branch}")
        if global_top_n > 0:
            _add_top_rows(reasons, eligible, int(global_top_n), "global")
        if not reasons:
            _add_top_rows(reasons, frame, 1, "fallback")
        selected = frame.loc[
            frame["_candidate_reservoir_input_index"].isin(reasons.keys())
        ].copy()
        selected["candidate_reservoir_reasons"] = selected[
            "_candidate_reservoir_input_index"
        ].map(lambda idx: ",".join(sorted(reasons[int(idx)])))
        selected = selected.sort_values(
            ["candidate_reservoir_score", "source", "candidate_branch"],
            ascending=[False, True, True],
        )
        if max_candidates_per_frame is not None and max_candidates_per_frame > 0:
            selected = selected.head(int(max_candidates_per_frame)).copy()
        selected["candidate_reservoir_rank"] = np.arange(1, len(selected) + 1, dtype=int)
        selected_frames.append(selected)

    reservoir = pd.concat(selected_frames, ignore_index=True)
    return reservoir.drop(columns=["_candidate_reservoir_input_index"])


def build_reservoir_summary(candidates: pd.DataFrame, reservoir: pd.DataFrame) -> dict[str, Any]:
    """Build a compact JSON-serializable reservoir summary."""

    input_counts = _frame_counts(candidates)
    reservoir_counts = _frame_counts(reservoir)
    return {
        "input_candidate_rows": int(len(candidates)),
        "reservoir_candidate_rows": int(len(reservoir)),
        "input_frame_count": int(len(input_counts)),
        "reservoir_frame_count": int(len(reservoir_counts)),
        "input_candidates_per_frame_mean": _safe_mean(input_counts),
        "reservoir_candidates_per_frame_mean": _safe_mean(reservoir_counts),
        "reservoir_candidates_per_frame_p95": _safe_quantile(reservoir_counts, 0.95),
        "reservoir_candidates_per_frame_max": _safe_max(reservoir_counts),
        "source_counts": _value_counts(reservoir, "source"),
        "candidate_branch_counts": _value_counts(reservoir, "candidate_branch"),
        "reservoir_reason_counts": _reason_counts(reservoir),
    }


def write_reservoir_outputs(
    reservoir: pd.DataFrame,
    *,
    output_csv: Path,
    summary_json: Path | None = None,
    input_candidates: pd.DataFrame | None = None,
) -> None:
    """Write reservoir CSV and optional summary JSON."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    reservoir.to_csv(output_csv, index=False)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = build_reservoir_summary(
            input_candidates if input_candidates is not None else reservoir,
            reservoir,
        )
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        required=True,
        help="candidate CSV, optionally as branch_name=/path/to/candidates.csv",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--top-per-source", type=int, default=3)
    parser.add_argument("--top-per-branch", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=40)
    parser.add_argument("--max-candidates-per-frame", type=int)
    parser.add_argument("--score-column", action="append", default=[])
    parser.add_argument("--score-floor-quantile", type=float)
    args = parser.parse_args(argv)

    candidates = load_candidate_inputs(args.candidate_csv)
    score_columns = tuple(args.score_column) if args.score_column else _DEFAULT_SCORE_COLUMNS
    reservoir = build_candidate_reservoir(
        candidates,
        top_per_source=args.top_per_source,
        top_per_branch=args.top_per_branch,
        global_top_n=args.global_top_n,
        max_candidates_per_frame=args.max_candidates_per_frame,
        score_columns=score_columns,
        score_floor_quantile=args.score_floor_quantile,
    )
    write_reservoir_outputs(
        reservoir,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        input_candidates=candidates,
    )
    print(f"candidate_reservoir_csv={args.output_csv}")
    if args.summary_json is not None:
        print(f"candidate_reservoir_summary_json={args.summary_json}")
    return 0


def _split_candidate_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        branch, path = spec.split("=", 1)
        branch = _sanitize_branch(branch)
        return branch, Path(path)
    path = Path(spec)
    return _sanitize_branch(path.stem), path


def _sanitize_branch(value: str) -> str:
    branch = str(value).strip().replace(" ", "_")
    return branch or "candidate_branch"


def _validate_required_columns(rows: pd.DataFrame, path: Path) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"candidate CSV {path} missing required columns: {missing}")


def _candidate_scores(rows: pd.DataFrame, score_columns: Sequence[str]) -> pd.Series:
    scores = pd.Series(np.nan, index=rows.index, dtype=float)
    for column in score_columns:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce")
        scores = scores.where(scores.notna(), values)
    return scores.fillna(0.0).astype(float)


def _add_rows(reasons: dict[int, set[str]], rows: pd.DataFrame, reason: str) -> None:
    for idx in rows["_candidate_reservoir_input_index"].to_numpy(int):
        reasons.setdefault(int(idx), set()).add(reason)


def _add_top_rows(
    reasons: dict[int, set[str]],
    rows: pd.DataFrame,
    count: int,
    reason: str,
) -> None:
    if rows.empty or count <= 0:
        return
    ranked = rows.sort_values(
        ["candidate_reservoir_score", "source", "candidate_branch"],
        ascending=[False, True, True],
    ).head(int(count))
    _add_rows(reasons, ranked, reason)


def _frame_counts(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        return pd.Series(dtype=float)
    return rows.groupby(["sequence_id", "time_s"], dropna=False).size().astype(float)


def _value_counts(rows: pd.DataFrame, column: str) -> dict[str, int]:
    if rows.empty or column not in rows.columns:
        return {}
    return {str(key): int(value) for key, value in rows[column].value_counts(dropna=False).items()}


def _reason_counts(rows: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if rows.empty or "candidate_reservoir_reasons" not in rows.columns:
        return counts
    for value in rows["candidate_reservoir_reasons"].fillna("").astype(str):
        for reason in value.split(","):
            if reason:
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def _safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _safe_quantile(values: Iterable[float], quantile: float) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.quantile(arr, quantile)) if arr.size else float("nan")


def _safe_max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
