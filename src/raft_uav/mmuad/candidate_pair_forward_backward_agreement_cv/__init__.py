"""Fold-integrity guard for agreement-adaptive pair-prior train CV.

The maintained implementation remains in the sibling
``candidate_pair_forward_backward_agreement_cv.py`` module. This package loads
that implementation under a private name, re-exports its API, and tightens fold
eligibility so every expected holdout sequence contributes exactly one finite
metric row.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = (
    Path(__file__).resolve().parent.parent
    / "candidate_pair_forward_backward_agreement_cv.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._candidate_pair_forward_backward_agreement_cv_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load agreement-pair CV implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

AgreementPairCVConfig = _IMPL.AgreementPairCVConfig

for _name in dir(_IMPL):
    if not _name.startswith("_"):
        globals().setdefault(_name, getattr(_IMPL, _name))


def aggregate_agreement_pair_cv_folds(
    fold_summary: pd.DataFrame,
    *,
    cv_config: AgreementPairCVConfig | None = None,
    expected_sequence_count: int | None = None,
) -> pd.DataFrame:
    """Aggregate folds while requiring one finite row per holdout sequence."""

    cv_cfg = cv_config or AgreementPairCVConfig()
    _IMPL._validate_cv_config(cv_cfg)
    rows = pd.DataFrame(fold_summary).copy()
    if rows.empty:
        return rows
    metric = cv_cfg.selection_metric
    if metric not in rows.columns:
        raise ValueError(f"fold summary missing selection metric {metric!r}")
    if "holdout_sequence_id" not in rows.columns:
        raise ValueError("fold summary missing holdout_sequence_id")

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
        finite_mask = np.isfinite(values.to_numpy(float))
        finite = values.loc[finite_mask]
        sequence_ids = group["holdout_sequence_id"].astype(str)
        valid_sequence_ids = sequence_ids.loc[finite_mask]
        observed_sequence_count = int(sequence_ids.nunique())
        valid_sequence_count = int(valid_sequence_ids.nunique())
        duplicate_fold_count = int(len(sequence_ids) - observed_sequence_count)
        expected = (
            int(expected_sequence_count)
            if expected_sequence_count is not None
            else observed_sequence_count
        )
        mean = float(finite.mean()) if not finite.empty else float("nan")
        tail = (
            float(finite.quantile(cv_cfg.tail_quantile))
            if not finite.empty
            else float("nan")
        )
        eligible = bool(
            expected > 0
            and len(group) == expected
            and len(finite) == expected
            and observed_sequence_count == expected
            and valid_sequence_count == expected
            and duplicate_fold_count == 0
        )
        record = {column: group.iloc[0][column] for column in config_columns}
        record.update(
            {
                "fold_count": int(len(group)),
                "valid_fold_count": int(len(finite)),
                "holdout_sequence_count": observed_sequence_count,
                "valid_holdout_sequence_count": valid_sequence_count,
                "duplicate_fold_count": duplicate_fold_count,
                "eligible": eligible,
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
            if eligible and np.isfinite(mean) and np.isfinite(tail)
            else float("inf")
        )
        records.append(record)

    aggregate = pd.DataFrame.from_records(records)
    return aggregate.sort_values(
        ["eligible", "risk_score", f"{metric}_mean", f"{metric}_std", "grid_label"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)


_IMPL.aggregate_agreement_pair_cv_folds = aggregate_agreement_pair_cv_folds

__all__ = sorted(name for name in dir(_IMPL) if not name.startswith("_"))
