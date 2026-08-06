"""Prevent sequence-gate feature-imputation leakage across data splits.

The maintained implementation lives in the sibling
``track5_sequence_gate_fit.py`` module. This package preserves the public import
path while ensuring that LOSO and apply-set feature imputation is fitted only on
the corresponding training rows.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


_IMPL_PATH = Path(__file__).resolve().parent.parent / "track5_sequence_gate_fit.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._track5_sequence_gate_fit_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(
        f"cannot load Track 5 sequence-gate implementation from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)
_ORIGINAL_MAIN = _IMPL.main
_make_model = _IMPL._make_model
_weight_table = _IMPL._weight_table


def main(argv: list[str] | None = None) -> int:
    """Run the legacy CLI with this compatibility package's pandas boundary."""

    original_pandas = _IMPL.pd
    _IMPL.pd = globals()["pd"]
    try:
        return _ORIGINAL_MAIN(argv)
    finally:
        _IMPL.pd = original_pandas


def _raw_feature_matrix(
    rows: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Return numeric feature values without split-dependent imputation."""

    matrix = (
        pd.DataFrame(rows)[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float, copy=True)
    )
    if matrix.ndim != 2:
        matrix = matrix.reshape(len(rows), -1)
    return matrix


def _feature_imputation_values(
    rows: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Fit one finite median per feature using only ``rows``."""

    matrix = _raw_feature_matrix(rows, feature_columns)
    values: list[float] = []
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        finite = column[np.isfinite(column)]
        values.append(float(np.median(finite)) if finite.size else 0.0)
    return np.asarray(values, dtype=float)


def _feature_matrix(
    rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    imputation_values: np.ndarray | None = None,
) -> np.ndarray:
    """Return a finite feature matrix using fitted or explicitly supplied medians."""

    matrix = _raw_feature_matrix(rows, feature_columns)
    if imputation_values is None:
        fill_values = _feature_imputation_values(rows, feature_columns)
    else:
        fill_values = np.asarray(imputation_values, dtype=float)
        if (
            fill_values.ndim != 1
            or len(fill_values) != matrix.shape[1]
            or not np.isfinite(fill_values).all()
        ):
            raise ValueError(
                "feature imputation values must contain one finite value per feature"
            )

    row_indices, column_indices = np.where(~np.isfinite(matrix))
    matrix[row_indices, column_indices] = fill_values[column_indices]
    return matrix


def _predict_loso_weights(
    model_name: str,
    rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    min_weight: float,
    max_weight: float,
) -> pd.DataFrame:
    """Predict each sequence using preprocessing fitted without that sequence."""

    work = pd.DataFrame(rows).copy().reset_index(drop=True)
    if len(work) < 2:
        raise ValueError(
            "LOSO sequence-gate prediction requires at least two sequences"
        )

    predictions: list[dict[str, Any]] = []
    for index, row in work.iterrows():
        train_rows = work.drop(index=index)
        fill_values = _feature_imputation_values(train_rows, feature_columns)
        model = _make_model(model_name, random_state=random_state)
        model.fit(
            _feature_matrix(
                train_rows,
                feature_columns,
                imputation_values=fill_values,
            ),
            train_rows["oracle_weight"],
        )
        held_out = pd.DataFrame([row])
        value = float(
            model.predict(
                _feature_matrix(
                    held_out,
                    feature_columns,
                    imputation_values=fill_values,
                )
            )[0]
        )
        predictions.append(
            {
                "sequence_id": str(row["sequence_id"]),
                "blend_weight": value,
            }
        )

    return _weight_table(
        pd.Series([row["sequence_id"] for row in predictions]),
        np.asarray([row["blend_weight"] for row in predictions], dtype=float),
        min_weight=min_weight,
        max_weight=max_weight,
    )


def _predict_apply_weights(
    model_name: str,
    train_rows: pd.DataFrame,
    apply_rows: pd.DataFrame,
    feature_columns: list[str],
    *,
    random_state: int,
    min_weight: float,
    max_weight: float,
) -> pd.DataFrame:
    """Predict apply rows using preprocessing fitted only on training rows."""

    apply_frame = pd.DataFrame(apply_rows)
    if apply_frame.empty:
        return pd.DataFrame(columns=["sequence_id", "blend_weight"])

    train_frame = pd.DataFrame(train_rows)
    if train_frame.empty:
        raise ValueError(
            "apply sequence-gate prediction requires non-empty training rows"
        )

    fill_values = _feature_imputation_values(train_frame, feature_columns)
    model = _make_model(model_name, random_state=random_state)
    model.fit(
        _feature_matrix(
            train_frame,
            feature_columns,
            imputation_values=fill_values,
        ),
        train_frame["oracle_weight"],
    )
    predicted = model.predict(
        _feature_matrix(
            apply_frame,
            feature_columns,
            imputation_values=fill_values,
        )
    )
    return _weight_table(
        apply_frame["sequence_id"],
        np.asarray(predicted, dtype=float),
        min_weight=min_weight,
        max_weight=max_weight,
    )


_IMPL.main = main
_IMPL._raw_feature_matrix = _raw_feature_matrix
_IMPL._feature_imputation_values = _feature_imputation_values
_IMPL._feature_matrix = _feature_matrix
_IMPL._predict_loso_weights = _predict_loso_weights
_IMPL._predict_apply_weights = _predict_apply_weights

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["main"] = main
globals()["_ORIGINAL_MAIN"] = _ORIGINAL_MAIN
globals()["_raw_feature_matrix"] = _raw_feature_matrix
globals()["_feature_imputation_values"] = _feature_imputation_values
globals()["_feature_matrix"] = _feature_matrix
globals()["_predict_loso_weights"] = _predict_loso_weights
globals()["_predict_apply_weights"] = _predict_apply_weights

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
