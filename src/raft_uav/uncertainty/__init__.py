from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "uncertainty.py"
_LEGACY_NAME = "_raft_uav_uncertainty_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load {_LEGACY_PATH}")
_legacy = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _legacy
_SPEC.loader.exec_module(_legacy)

VarianceHead = _legacy.VarianceHead
HeteroscedasticUncertaintyModel = _legacy.HeteroscedasticUncertaintyModel
_original_variance_head_init = VarianceHead.__init__
_original_model_init = HeteroscedasticUncertaintyModel.__init__
_original_model_from_dict = HeteroscedasticUncertaintyModel.from_dict.__func__
_original_aligned_residuals = _legacy._aligned_residuals
_original_fit_heteroscedastic_uncertainty_model = (
    _legacy.fit_heteroscedastic_uncertainty_model
)
_MISSING_SEQUENCE_KEYS = frozenset({"nan", "none", "<na>", "nat"})


def _finite_scalar(value: object, field: str) -> float:
    scalar = value
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{field} must be a finite number")
        scalar = value.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(f"{field} must be a finite number")
    if isinstance(scalar, (complex, np.complexfloating)):
        raise ValueError(f"{field} must be a finite number")
    try:
        number = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _nonnegative_fit_control(value: object, field: str) -> float:
    number = _finite_scalar(value, field)
    if number < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def _exact_integer_scalar(value: object, field: str) -> int:
    """Return an exact integer scalar without truncation or Boolean coercion."""

    error = f"{field} must be an exact integer scalar"
    if np.ma.is_masked(value) or isinstance(value, (bool, np.bool_)):
        raise ValueError(error)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc
    if scalar.ndim != 0 or np.iscomplexobj(scalar):
        raise ValueError(error)
    try:
        item = scalar.item()
        if np.ma.is_masked(item) or isinstance(
            item,
            (bool, np.bool_, complex, np.complexfloating),
        ):
            raise ValueError(error)
        number = float(item)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(error)
    return int(number)


def _validate_variance_head(head) -> None:
    source = head.source
    if source not in _legacy.SOURCE_DIMS:
        raise ValueError(f"unknown uncertainty source {source!r}")
    if head.dimension not in _legacy.SOURCE_DIMS[source]:
        raise ValueError(
            f"dimension {head.dimension!r} is invalid for uncertainty source {source!r}"
        )

    feature_names = tuple(head.feature_names)
    allowed_features = (
        set(_legacy.RF_FEATURES) if source == "rf" else set(_legacy.RADAR_FEATURES)
    )
    if not feature_names:
        raise ValueError("variance head must contain at least one feature")
    if not all(isinstance(name, str) and name for name in feature_names):
        raise ValueError("variance head feature names must be non-empty strings")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("variance head feature names must be unique")
    unknown_features = sorted(set(feature_names) - allowed_features)
    if unknown_features:
        raise ValueError(f"unknown {source} uncertainty features: {unknown_features}")

    try:
        coefficients = np.asarray(head.coefficients, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("variance head coefficients must be finite numbers") from exc
    if coefficients.size != len(feature_names):
        raise ValueError("feature/coefficient dimension mismatch")
    if not np.isfinite(coefficients).all():
        raise ValueError("variance head coefficients must be finite numbers")

    min_std_m = _finite_scalar(head.min_std_m, "min_std_m")
    max_std_m = _finite_scalar(head.max_std_m, "max_std_m")
    if min_std_m <= 0.0:
        raise ValueError("min_std_m must be positive")
    if max_std_m < min_std_m:
        raise ValueError("max_std_m must be greater than or equal to min_std_m")


def _validated_variance_head_init(
    self,
    source,
    dimension,
    feature_names,
    coefficients,
    min_std_m,
    max_std_m,
    training_rows,
):
    _original_variance_head_init(
        self,
        source,
        dimension,
        feature_names,
        coefficients,
        min_std_m,
        max_std_m,
        training_rows,
    )
    _validate_variance_head(self)


def _validated_model_init(self, heads, metadata):
    _original_model_init(self, heads, metadata)
    if not isinstance(self.metadata, Mapping):
        raise ValueError("uncertainty model metadata must be a mapping")
    seen: set[tuple[str, str]] = set()
    for head in self.heads:
        if not isinstance(head, VarianceHead):
            raise ValueError("uncertainty model heads must be VarianceHead instances")
        key = (head.source, head.dimension)
        if key in seen:
            raise ValueError(
                "duplicate uncertainty variance head for "
                f"{head.source!r}/{head.dimension!r}"
            )
        seen.add(key)


def _validated_model_from_dict(cls, payload):
    """Load only artifacts that declare an exact supported schema version."""

    if not isinstance(payload, Mapping):
        raise ValueError("uncertainty model payload must be a mapping")
    schema_value = payload.get("schema_version", 0)
    schema_version = _exact_integer_scalar(schema_value, "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported uncertainty schema {schema_value!r}")
    return _original_model_from_dict(cls, payload)


def _sequence_keys(values: pd.Series) -> pd.Series:
    """Return trimmed sequence identifiers while preserving missing values."""

    keys = pd.Series(values, index=values.index, dtype="string").str.strip()
    missing = keys.isna() | keys.eq("") | keys.str.lower().isin(
        _MISSING_SEQUENCE_KEYS
    )
    return keys.mask(missing)


def _aligned_residuals(
    frame: pd.DataFrame,
    truth: pd.DataFrame,
    *,
    max_time_delta_s: float,
) -> pd.DataFrame:
    """Align residuals within sequence boundaries before nearest-time matching."""

    if "sequence_id" not in frame.columns or "sequence_id" not in truth.columns:
        return _original_aligned_residuals(
            frame,
            truth,
            max_time_delta_s=max_time_delta_s,
        )

    frame_keys = _sequence_keys(frame["sequence_id"])
    truth_keys = _sequence_keys(truth["sequence_id"])
    order_column = "__raft_uav_uncertainty_alignment_order__"
    while order_column in frame.columns:
        order_column += "_"

    positioned = frame.copy()
    positioned[order_column] = np.arange(len(positioned), dtype=int)
    blocks: list[pd.DataFrame] = []
    for sequence_id in pd.unique(frame_keys.dropna()):
        frame_mask = frame_keys.eq(sequence_id).fillna(False)
        truth_mask = truth_keys.eq(sequence_id).fillna(False)
        sequence_truth = truth.loc[truth_mask]
        if sequence_truth.empty:
            continue
        block = _original_aligned_residuals(
            positioned.loc[frame_mask],
            sequence_truth,
            max_time_delta_s=max_time_delta_s,
        )
        if not block.empty:
            blocks.append(block)

    if not blocks:
        return frame.iloc[0:0].copy()

    aligned = pd.concat(blocks, ignore_index=False)
    aligned = aligned.sort_values(order_column, kind="mergesort")
    return aligned.drop(columns=order_column).reset_index(drop=True)


def fit_heteroscedastic_uncertainty_model(
    *,
    rf: pd.DataFrame | None,
    radar: pd.DataFrame | None,
    truth: pd.DataFrame,
    ridge_lambda: float = 1.0,
    max_time_delta_s: float = 2.0,
    min_std_m: Mapping[str, Mapping[str, float]] | None = None,
    max_std_m: Mapping[str, Mapping[str, float]] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> HeteroscedasticUncertaintyModel:
    """Fit uncertainty heads after validating optimization controls."""

    validated_ridge_lambda = _nonnegative_fit_control(
        ridge_lambda,
        "ridge_lambda",
    )
    validated_max_time_delta_s = _nonnegative_fit_control(
        max_time_delta_s,
        "max_time_delta_s",
    )
    return _original_fit_heteroscedastic_uncertainty_model(
        rf=rf,
        radar=radar,
        truth=truth,
        ridge_lambda=validated_ridge_lambda,
        max_time_delta_s=validated_max_time_delta_s,
        min_std_m=min_std_m,
        max_std_m=max_std_m,
        metadata=metadata,
    )


VarianceHead.__init__ = _validated_variance_head_init
HeteroscedasticUncertaintyModel.__init__ = _validated_model_init
HeteroscedasticUncertaintyModel.from_dict = classmethod(
    _validated_model_from_dict
)
_legacy._aligned_residuals = _aligned_residuals
_legacy.fit_heteroscedastic_uncertainty_model = (
    fit_heteroscedastic_uncertainty_model
)

for _name in dir(_legacy):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_legacy, _name)
globals()["VarianceHead"] = VarianceHead
globals()["HeteroscedasticUncertaintyModel"] = HeteroscedasticUncertaintyModel
globals()["_sequence_keys"] = _sequence_keys
globals()["_aligned_residuals"] = _aligned_residuals
globals()["_finite_scalar"] = _finite_scalar
globals()["_nonnegative_fit_control"] = _nonnegative_fit_control
globals()["_exact_integer_scalar"] = _exact_integer_scalar
globals()["_validated_model_from_dict"] = _validated_model_from_dict
globals()["fit_heteroscedastic_uncertainty_model"] = (
    fit_heteroscedastic_uncertainty_model
)
__doc__ = _legacy.__doc__
__all__ = [_name for _name in dir(_legacy) if not _name.startswith("__")]
