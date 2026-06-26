#!/usr/bin/env python
"""Sweep truth-free MMUAD template-aligned branch reservoir settings.

The CVPR UG2+/MMUAD Track 5 test split is driven by an official
Sequence/Timestamp template.  This helper evaluates reservoir-preparation knobs
without reading pose truth so candidate pools can be chosen by coverage,
branch/source diversity, and candidate budget before downstream tracker or
mixture-MAP runs.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPT_ROOT = Path(__file__).resolve().parent
for root in (SRC_ROOT, SCRIPT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from mmuad_template_branch_reservoir import (  # noqa: E402
    SCORE_NORMALIZATION_CHOICES,
    load_branch_candidate_inputs,
    parse_candidate_input,
    write_template_branch_reservoir_artifacts,
)
from raft_uav.mmuad.submission import load_official_track5_template_file  # noqa: E402

SUMMARY_CSV = "mmuad_template_reservoir_sweep_summary.csv"
SUMMARY_JSON = "mmuad_template_reservoir_sweep_summary.json"


def run_template_reservoir_sweep(
    *,
    candidates: pd.DataFrame,
    template: pd.DataFrame,
    output_dir: Path,
    max_time_delta_s_values: Iterable[float],
    score_normalizations: Iterable[str] = ("none",),
    min_candidates_per_template_values: Iterable[int] = (0,),
    fallback_max_time_delta_s_values: Iterable[float | None] = (None,),
    per_source_top_n: int = 3,
    per_branch_top_n: int = 3,
    global_top_n: int = 20,
    score_column: str = "ranker_score",
    score_floor_quantile: float | None = None,
    max_candidates_per_template: int | None = None,
) -> pd.DataFrame:
    """Run a truth-free reservoir sweep and return one row per variant."""

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for delta_s, normalization, min_count, fallback_delta_s in itertools.product(
        _unique_float_values(max_time_delta_s_values),
        _unique_text_values(score_normalizations),
        _unique_int_values(min_candidates_per_template_values),
        _unique_optional_float_values(fallback_max_time_delta_s_values),
    ):
        variant_label = _variant_label(
            delta_s=delta_s,
            score_normalization=normalization,
            min_candidates_per_template=min_count,
            fallback_delta_s=fallback_delta_s,
        )
        variant_dir = output_dir / variant_label
        paths = write_template_branch_reservoir_artifacts(
            candidates,
            template,
            output_dir=variant_dir,
            max_time_delta_s=float(delta_s),
            per_source_top_n=int(per_source_top_n),
            per_branch_top_n=int(per_branch_top_n),
            global_top_n=int(global_top_n),
            score_column=str(score_column),
            score_floor_quantile=score_floor_quantile,
            max_candidates_per_template=max_candidates_per_template,
            score_normalization=str(normalization),
            min_candidates_per_template=int(min_count),
            fallback_max_time_delta_s=fallback_delta_s,
            provenance={"sweep_variant_label": variant_label},
        )
        frame_summary = pd.read_csv(paths["frame_summary_csv"])
        branch_summary = pd.read_csv(paths["branch_summary_csv"])
        reservoir = pd.read_csv(paths["reservoir_csv"])
        records.append(
            _summary_record(
                variant_label=variant_label,
                max_time_delta_s=delta_s,
                score_normalization=normalization,
                min_candidates_per_template=min_count,
                fallback_max_time_delta_s=fallback_delta_s,
                frame_summary=frame_summary,
                branch_summary=branch_summary,
                reservoir=reservoir,
                paths=paths,
            )
        )
    summary = pd.DataFrame.from_records(records)
    summary = _sort_summary(summary)
    summary.to_csv(output_dir / SUMMARY_CSV, index=False)
    payload = {"rows": summary.to_dict(orient="records")}
    summary_json = json.dumps(payload, indent=2)
    (output_dir / SUMMARY_JSON).write_text(summary_json, encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-csv", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        action="append",
        default=[],
        metavar="BRANCH=PATH",
        help="candidate CSV to include; may be repeated; plain PATH uses file stem as branch",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-time-delta-s",
        action="append",
        default=[],
        help="candidate/template time windows in seconds; repeat or comma-separate values",
    )
    parser.add_argument(
        "--score-normalization",
        action="append",
        choices=SCORE_NORMALIZATION_CHOICES,
        default=[],
        help="score normalization modes to sweep; repeat for multiple modes",
    )
    parser.add_argument(
        "--min-candidates-per-template",
        action="append",
        default=[],
        help="minimum retained candidates per template row; repeat or comma-separate values",
    )
    parser.add_argument(
        "--fallback-max-time-delta-s",
        action="append",
        default=[],
        help="fallback time windows; use none for unlimited; repeat or comma-separate values",
    )
    parser.add_argument("--per-source-top-n", type=int, default=3)
    parser.add_argument("--per-branch-top-n", type=int, default=3)
    parser.add_argument("--global-top-n", type=int, default=20)
    parser.add_argument("--score-column", default="ranker_score")
    parser.add_argument("--score-floor-quantile", type=float)
    parser.add_argument("--max-candidates-per-template", type=int)
    args = parser.parse_args(argv)

    if not args.candidate_csv:
        parser.error("provide at least one --candidate-csv BRANCH=PATH")
    inputs = tuple(parse_candidate_input(value) for value in args.candidate_csv)
    candidates = load_branch_candidate_inputs(inputs)
    template = load_official_track5_template_file(args.template_csv)
    summary = run_template_reservoir_sweep(
        candidates=candidates,
        template=template,
        output_dir=args.output_dir,
        max_time_delta_s_values=_parse_float_list(
            args.max_time_delta_s or ["0.25,0.5,1.0"]
        ),
        score_normalizations=tuple(args.score_normalization) or ("none",),
        min_candidates_per_template_values=_parse_int_list(
            args.min_candidates_per_template or ["0"]
        ),
        fallback_max_time_delta_s_values=_parse_optional_float_list(
            args.fallback_max_time_delta_s or ["none"]
        ),
        per_source_top_n=int(args.per_source_top_n),
        per_branch_top_n=int(args.per_branch_top_n),
        global_top_n=int(args.global_top_n),
        score_column=str(args.score_column),
        score_floor_quantile=args.score_floor_quantile,
        max_candidates_per_template=args.max_candidates_per_template,
    )
    print("mmuad_template_reservoir_sweep=ok")
    print(f"summary_csv={args.output_dir / SUMMARY_CSV}")
    print(f"variant_count={len(summary)}")
    return 0


def _summary_record(
    *,
    variant_label: str,
    max_time_delta_s: float,
    score_normalization: str,
    min_candidates_per_template: int,
    fallback_max_time_delta_s: float | None,
    frame_summary: pd.DataFrame,
    branch_summary: pd.DataFrame,
    reservoir: pd.DataFrame,
    paths: dict[str, Path],
) -> dict[str, object]:
    reservoir_count = pd.to_numeric(frame_summary.get("reservoir_count"), errors="coerce")
    window_count = pd.to_numeric(frame_summary.get("candidate_count_window"), errors="coerce")
    fallback_count = pd.to_numeric(frame_summary.get("fallback_retained_count"), errors="coerce")
    branch_count = pd.to_numeric(frame_summary.get("branch_count_reservoir"), errors="coerce")
    source_count = pd.to_numeric(frame_summary.get("source_count_reservoir"), errors="coerce")
    covered = reservoir_count.fillna(0) > 0
    target_min = max(int(min_candidates_per_template), 1)
    enough_candidates = reservoir_count.fillna(0) >= target_min
    return {
        "variant_label": variant_label,
        "max_time_delta_s": float(max_time_delta_s),
        "score_normalization": str(score_normalization),
        "min_candidates_per_template": int(min_candidates_per_template),
        "fallback_max_time_delta_s": fallback_max_time_delta_s,
        "template_rows": int(len(frame_summary)),
        "reservoir_rows": int(len(reservoir)),
        "covered_template_rows": int(covered.sum()),
        "coverage_fraction": float(covered.mean()) if len(covered) else 0.0,
        "template_rows_with_min_candidates": int(enough_candidates.sum()),
        "min_candidate_coverage_fraction": float(enough_candidates.mean())
        if len(enough_candidates)
        else 0.0,
        "mean_window_candidate_count": _mean(window_count),
        "mean_reservoir_count": _mean(reservoir_count),
        "p95_reservoir_count": _percentile(reservoir_count, 95),
        "max_reservoir_count": _max(reservoir_count),
        "mean_reservoir_branch_count": _mean(branch_count),
        "mean_reservoir_source_count": _mean(source_count),
        "fallback_rows": int(fallback_count.fillna(0).sum()) if len(fallback_count) else 0,
        "branch_summary_rows": int(len(branch_summary)),
        "reservoir_csv": str(paths["reservoir_csv"]),
        "frame_summary_csv": str(paths["frame_summary_csv"]),
        "branch_summary_csv": str(paths["branch_summary_csv"]),
        "provenance_json": str(paths["provenance_json"]),
    }


def _sort_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    columns = [
        "min_candidate_coverage_fraction",
        "coverage_fraction",
        "mean_reservoir_branch_count",
        "mean_reservoir_source_count",
        "mean_reservoir_count",
    ]
    existing = [column for column in columns if column in summary.columns]
    return summary.sort_values(existing, ascending=[False] * len(existing)).reset_index(
        drop=True
    )


def _variant_label(
    *,
    delta_s: float,
    score_normalization: str,
    min_candidates_per_template: int,
    fallback_delta_s: float | None,
) -> str:
    fallback = "none" if fallback_delta_s is None else _float_label(fallback_delta_s)
    return (
        f"dt{_float_label(delta_s)}_score{score_normalization}_"
        f"min{int(min_candidates_per_template)}_fb{fallback}"
    )


def _parse_float_list(values: Iterable[str]) -> tuple[float, ...]:
    parsed: list[float] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                parsed.append(float(item))
    return _unique_float_values(parsed)


def _parse_int_list(values: Iterable[str]) -> tuple[int, ...]:
    parsed: list[int] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip()
            if item:
                parsed.append(int(item))
    return _unique_int_values(parsed)


def _parse_optional_float_list(values: Iterable[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        for item in str(value).replace(";", ",").split(","):
            item = item.strip().lower()
            if not item:
                continue
            parsed.append(None if item in {"none", "null", "unlimited"} else float(item))
    return tuple(parsed or [None])


def _unique_float_values(values: Iterable[float]) -> tuple[float, ...]:
    return tuple(sorted({float(value) for value in values}))


def _unique_int_values(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


def _unique_text_values(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _unique_optional_float_values(values: Iterable[float | None]) -> tuple[float | None, ...]:
    seen: list[float | None] = []
    for value in values:
        normalized = None if value is None else float(value)
        if normalized not in seen:
            seen.append(normalized)
    return tuple(seen or [None])


def _float_label(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _finite_values(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    return numeric[np.isfinite(numeric)]


def _mean(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(np.mean(finite)) if len(finite) else np.nan


def _percentile(values: pd.Series, percentile: float) -> float:
    finite = _finite_values(values)
    return float(np.percentile(finite, percentile)) if len(finite) else np.nan


def _max(values: pd.Series) -> float:
    finite = _finite_values(values)
    return float(np.max(finite)) if len(finite) else np.nan


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
