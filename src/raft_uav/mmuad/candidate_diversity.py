"""Diversity-aware pruning for MMUAD candidate reservoirs.

The branch-preserving reservoir can still waste its per-frame budget on many
near-duplicate candidates. This module applies score-aware spatial suppression
while protecting candidates selected for explicit source/branch reasons.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def diversify_candidate_reservoir(
    rows: pd.DataFrame,
    *,
    radius_m: float = 1.0,
    max_candidates_per_frame: int = 40,
    score_column: str = "candidate_reservoir_score",
    preserve_protected: bool = True,
) -> pd.DataFrame:
    """Keep spatially diverse candidates in every sequence/time frame.

    Protected rows are reserved separately. Remaining rows are considered by
    score and accepted only when they are at least ``radius_m`` away from every
    previously selected ordinary row. This prevents a low-score protected row
    from suppressing a stronger nearby hypothesis while preserving explicit
    source/branch selections under the frame cap.
    """

    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame
    required = {"sequence_id", "time_s", "x_m", "y_m", "z_m"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate reservoir missing required columns: {missing}")
    if score_column not in frame.columns:
        frame[score_column] = pd.to_numeric(frame.get("confidence", 0.0), errors="coerce")
    frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0)
    if "candidate_reservoir_protected" not in frame.columns:
        frame["candidate_reservoir_protected"] = False

    try:
        radius = float(radius_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("radius_m must be finite and non-negative") from exc
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius_m must be finite and non-negative")

    try:
        cap = int(max_candidates_per_frame)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("max_candidates_per_frame must be a non-negative integer") from exc
    if cap != max_candidates_per_frame or cap < 0:
        raise ValueError("max_candidates_per_frame must be a non-negative integer")

    outputs: list[pd.DataFrame] = []
    for _, group in frame.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        group = group.copy()
        protected = group["candidate_reservoir_protected"].fillna(False).astype(bool)
        if not preserve_protected:
            protected = pd.Series(False, index=group.index, dtype=bool)

        protected_order = group.loc[protected].sort_values(
            score_column,
            ascending=False,
            kind="mergesort",
        )
        protected_indices = [
            idx
            for idx, row in protected_order.iterrows()
            if np.isfinite(row[["x_m", "y_m", "z_m"]].to_numpy(float)).all()
        ]

        ordinary_order = group.loc[~protected].sort_values(
            score_column,
            ascending=False,
            kind="mergesort",
        )
        ordinary_indices: list[int] = []
        ordinary_xyz: list[np.ndarray] = []
        for idx, row in ordinary_order.iterrows():
            xyz = row[["x_m", "y_m", "z_m"]].to_numpy(float)
            if not np.isfinite(xyz).all():
                continue
            sufficiently_distinct = not ordinary_xyz or min(
                float(np.linalg.norm(xyz - prior)) for prior in ordinary_xyz
            ) >= radius
            if sufficiently_distinct:
                ordinary_indices.append(idx)
                ordinary_xyz.append(xyz)

        if cap > 0:
            selected = protected_indices[:cap]
            selected.extend(ordinary_indices[: max(cap - len(selected), 0)])
        else:
            selected = [*protected_indices, *ordinary_indices]
        out = group.loc[selected].copy()
        out = out.sort_values(score_column, ascending=False, kind="mergesort")
        out["candidate_diversity_rank"] = np.arange(1, len(out) + 1)
        out["candidate_diversity_radius_m"] = radius
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True) if outputs else frame.iloc[0:0].copy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--do-not-preserve-protected", action="store_true")
    args = parser.parse_args(argv)

    rows = pd.read_csv(args.input_csv)
    output = diversify_candidate_reservoir(
        rows,
        radius_m=args.radius_m,
        max_candidates_per_frame=args.max_candidates_per_frame,
        score_column=args.score_column,
        preserve_protected=not args.do_not_preserve_protected,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"candidate_diversity_csv={args.output_csv}")
    print(f"candidate_rows_before={len(rows)}")
    print(f"candidate_rows_after={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
