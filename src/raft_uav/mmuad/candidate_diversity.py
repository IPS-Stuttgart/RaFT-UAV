"""Diversity-aware pruning for MMUAD candidate reservoirs.

The branch-preserving reservoir can still waste its per-frame budget on many
near-duplicate candidates. This module applies score-aware spatial suppression
while protecting candidates selected for explicit source/branch reasons.

When a learned candidate-uncertainty column is supplied, the exclusion radius
is adapted per selected non-protected candidate. Precise candidates suppress a
wider local neighbourhood, while uncertain candidates keep a smaller exclusion
zone so nearby alternatives can survive for mixture-MAP inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

_TRUE_PROTECTED_TEXT = {"1", "true", "t", "yes", "y", "on"}
_FALSE_PROTECTED_TEXT = {
    "",
    "0",
    "false",
    "f",
    "no",
    "n",
    "off",
    "nan",
    "none",
    "null",
    "<na>",
}


def diversify_candidate_reservoir(
    rows: pd.DataFrame,
    *,
    radius_m: float = 1.0,
    max_candidates_per_frame: int = 40,
    score_column: str = "candidate_reservoir_score",
    preserve_protected: bool = True,
    uncertainty_column: str | None = None,
    uncertainty_reference_m: float = 10.0,
    uncertainty_exponent: float = 0.5,
    min_radius_scale: float = 0.25,
    max_radius_scale: float = 4.0,
) -> pd.DataFrame:
    """Keep spatially diverse candidates in every sequence/time frame.

    Protected rows are retained without suppressing other hypotheses. Remaining
    rows are considered by score and accepted only when they are outside every
    previously accepted non-protected candidate's exclusion radius.

    With ``uncertainty_column=None``, every suppressing candidate uses the fixed
    ``radius_m`` exclusion radius. With an uncertainty column, candidate ``i``
    uses

    ``radius_i = radius_m * clip((reference / sigma_i) ** exponent, min, max)``.

    Thus a low predicted error suppresses more local duplicates, while a high
    predicted error preserves nearby alternatives. Invalid or non-positive
    uncertainty values are imputed with the reference value and reported in the
    output diagnostics.
    """

    frame = pd.DataFrame(rows).copy().reset_index(drop=True)
    if frame.empty:
        return frame
    required = {"sequence_id", "time_s", "x_m", "y_m", "z_m"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"candidate reservoir missing required columns: {missing}")
    primary_scores = pd.to_numeric(
        frame.get(score_column, pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    fallback_scores = pd.to_numeric(
        frame.get("confidence", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    )
    primary_scores = primary_scores.where(
        np.isfinite(primary_scores.to_numpy(dtype=float))
    )
    fallback_scores = fallback_scores.where(
        np.isfinite(fallback_scores.to_numpy(dtype=float))
    )
    frame[score_column] = primary_scores.fillna(fallback_scores).fillna(0.0)
    if "candidate_reservoir_protected" not in frame.columns:
        frame["candidate_reservoir_protected"] = False

    radius = max(float(radius_m), 0.0)
    cap = max(int(max_candidates_per_frame), 1)
    effective_radius, uncertainty, imputed = _effective_candidate_radii(
        frame,
        base_radius_m=radius,
        uncertainty_column=uncertainty_column,
        uncertainty_reference_m=uncertainty_reference_m,
        uncertainty_exponent=uncertainty_exponent,
        min_radius_scale=min_radius_scale,
        max_radius_scale=max_radius_scale,
    )
    frame["_candidate_diversity_effective_radius_m"] = effective_radius
    if uncertainty_column is not None:
        frame["candidate_diversity_uncertainty_m"] = uncertainty
        frame["candidate_diversity_uncertainty_imputed"] = imputed

    outputs: list[pd.DataFrame] = []
    for _, group in frame.groupby(["sequence_id", "time_s"], sort=False, dropna=False):
        group = group.copy()
        if preserve_protected:
            priority = _protected_flag_series(group["candidate_reservoir_protected"])
        else:
            priority = pd.Series(False, index=group.index, dtype=bool)
        order = group.assign(_protected=priority).sort_values(
            ["_protected", score_column],
            ascending=[False, False],
            kind="mergesort",
        )
        selected: list[int] = []
        suppression_xyz: list[np.ndarray] = []
        suppression_radii: list[float] = []
        for idx, row in order.iterrows():
            is_protected = bool(row["_protected"])
            xyz = row[["x_m", "y_m", "z_m"]].to_numpy(float)
            if not np.isfinite(xyz).all():
                continue
            sufficiently_distinct = all(
                float(np.linalg.norm(xyz - prior_xyz)) >= prior_radius
                for prior_xyz, prior_radius in zip(
                    suppression_xyz,
                    suppression_radii,
                    strict=True,
                )
            )
            if is_protected or sufficiently_distinct:
                selected.append(idx)
                if not is_protected:
                    suppression_xyz.append(xyz)
                    suppression_radii.append(
                        float(row["_candidate_diversity_effective_radius_m"])
                    )
            if len(selected) >= cap:
                break
        out = group.loc[selected].copy()
        out = out.sort_values(score_column, ascending=False, kind="mergesort")
        out["candidate_diversity_rank"] = np.arange(1, len(out) + 1)
        out["candidate_diversity_radius_m"] = radius
        out["candidate_diversity_effective_radius_m"] = out.pop(
            "_candidate_diversity_effective_radius_m"
        )
        outputs.append(out)
    if not outputs:
        return frame.iloc[0:0].drop(
            columns=["_candidate_diversity_effective_radius_m"],
            errors="ignore",
        )
    return pd.concat(outputs, ignore_index=True)


def _protected_flag_series(values: pd.Series) -> pd.Series:
    return values.map(_parse_protected_flag).astype(bool)


def _parse_protected_flag(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_PROTECTED_TEXT:
            return True
        if text in _FALSE_PROTECTED_TEXT:
            return False
        raise ValueError(
            "candidate_reservoir_protected values must be boolean-like; "
            f"got {value!r}"
        )
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "candidate_reservoir_protected values must be boolean-like; "
            f"got {value!r}"
        ) from exc
    if not np.isfinite(number):
        return False
    return bool(number)


def _effective_candidate_radii(
    rows: pd.DataFrame,
    *,
    base_radius_m: float,
    uncertainty_column: str | None,
    uncertainty_reference_m: float,
    uncertainty_exponent: float,
    min_radius_scale: float,
    max_radius_scale: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    reference = float(uncertainty_reference_m)
    exponent = float(uncertainty_exponent)
    minimum = float(min_radius_scale)
    maximum = float(max_radius_scale)
    controls = {
        "uncertainty_reference_m": reference,
        "uncertainty_exponent": exponent,
        "min_radius_scale": minimum,
        "max_radius_scale": maximum,
    }
    non_finite = [name for name, value in controls.items() if not np.isfinite(value)]
    if non_finite:
        raise ValueError(f"non-finite uncertainty diversity controls: {non_finite}")
    if reference <= 0.0:
        raise ValueError("uncertainty_reference_m must be positive")
    if exponent < 0.0:
        raise ValueError("uncertainty_exponent must be non-negative")
    if minimum <= 0.0 or maximum <= 0.0:
        raise ValueError("radius scales must be positive")
    if minimum > maximum:
        raise ValueError("min_radius_scale must not exceed max_radius_scale")

    if uncertainty_column is None:
        uncertainty = pd.Series(reference, index=rows.index, dtype=float)
        imputed = pd.Series(False, index=rows.index, dtype=bool)
        radii = pd.Series(float(base_radius_m), index=rows.index, dtype=float)
        return radii, uncertainty, imputed
    if uncertainty_column not in rows.columns:
        raise ValueError(f"candidate reservoir missing uncertainty column: {uncertainty_column}")

    raw = pd.to_numeric(rows[uncertainty_column], errors="coerce")
    raw_values = raw.to_numpy(float)
    valid = np.isfinite(raw_values) & (raw_values > 0.0)
    imputed = pd.Series(~valid, index=rows.index, dtype=bool)
    uncertainty = raw.where(valid, reference).astype(float)
    scale = np.power(reference / uncertainty.to_numpy(float), exponent)
    scale = np.clip(scale, minimum, maximum)
    radii = pd.Series(float(base_radius_m) * scale, index=rows.index, dtype=float)
    return radii, uncertainty, imputed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--radius-m", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-frame", type=int, default=40)
    parser.add_argument("--score-column", default="candidate_reservoir_score")
    parser.add_argument("--do-not-preserve-protected", action="store_true")
    parser.add_argument("--uncertainty-column")
    parser.add_argument("--uncertainty-reference-m", type=float, default=10.0)
    parser.add_argument("--uncertainty-exponent", type=float, default=0.5)
    parser.add_argument("--min-radius-scale", type=float, default=0.25)
    parser.add_argument("--max-radius-scale", type=float, default=4.0)
    args = parser.parse_args(argv)

    rows = pd.read_csv(args.input_csv)
    output = diversify_candidate_reservoir(
        rows,
        radius_m=args.radius_m,
        max_candidates_per_frame=args.max_candidates_per_frame,
        score_column=args.score_column,
        preserve_protected=not args.do_not_preserve_protected,
        uncertainty_column=args.uncertainty_column,
        uncertainty_reference_m=args.uncertainty_reference_m,
        uncertainty_exponent=args.uncertainty_exponent,
        min_radius_scale=args.min_radius_scale,
        max_radius_scale=args.max_radius_scale,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(f"candidate_diversity_csv={args.output_csv}")
    print(f"candidate_rows_before={len(rows)}")
    print(f"candidate_rows_after={len(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
