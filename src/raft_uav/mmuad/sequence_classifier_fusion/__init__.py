"""Compatibility package validating sequence-classifier fusion inputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

_IMPL_PATH = Path(__file__).resolve().parent.parent / "sequence_classifier_fusion.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._sequence_classifier_fusion_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(
        "cannot load sequence-classifier fusion implementation "
        f"from {_IMPL_PATH}"
    )
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

FusionModelSpec = _IMPL.FusionModelSpec
FusionSelectionResult = _IMPL.FusionSelectionResult
_LEGACY_FUSE_SEQUENCE_PROBABILITIES = _IMPL.fuse_sequence_probabilities
_LEGACY_SELECT_TRAIN_SAFE_FUSION = _IMPL.select_train_safe_fusion
_LEGACY_MAIN = _IMPL.main


@dataclass(frozen=True)
class _SelectionModelSpec:
    """One model specification with a collision-free selection name."""

    method: str
    n_estimators: int
    max_depth: int | None
    random_state: int
    selection_name: str

    @property
    def name(self) -> str:
        return self.selection_name


def _distinct_model_specs(
    model_specs: list[FusionModelSpec],
) -> list[FusionModelSpec | _SelectionModelSpec]:
    """Give colliding model specifications stable, seed-aware identities."""

    specs = list(model_specs)
    name_counts = Counter(spec.name for spec in specs)
    distinct: list[FusionModelSpec | _SelectionModelSpec] = []
    for index, spec in enumerate(specs):
        if name_counts[spec.name] == 1:
            distinct.append(spec)
            continue
        distinct.append(
            _SelectionModelSpec(
                method=spec.method,
                n_estimators=spec.n_estimators,
                max_depth=spec.max_depth,
                random_state=spec.random_state,
                selection_name=(
                    f"{spec.name}_seed{spec.random_state}_candidate{index}"
                ),
            )
        )
    return distinct


def _validated_image_weight(value: Any, *, name: str = "image_weight") -> float:
    """Return a finite non-Boolean scalar fusion weight in the unit interval."""

    message = f"{name} must be a finite real scalar in [0, 1]"
    if isinstance(value, (bool, np.bool_)) or np.ma.is_masked(value):
        raise ValueError(message)
    try:
        scalar = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if scalar.ndim != 0 or scalar.dtype.kind in {"b", "c"}:
        raise ValueError(message)
    try:
        weight = float(scalar.item())
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError(message)
    return weight


def _validated_sequence_ids(rows: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Return rows with unique, non-missing opaque sequence identifiers."""

    out = pd.DataFrame(rows).copy()
    if "sequence_id" not in out.columns:
        raise ValueError(f"{name} must contain a sequence_id column")

    sequence = out["sequence_id"].astype("string").str.strip()
    missing_sequence = sequence.isna() | sequence.eq("")
    if bool(missing_sequence.any()):
        raise ValueError(f"{name} contains missing sequence_id values")

    sequence = sequence.astype(str)
    duplicates = sequence.loc[sequence.duplicated(keep=False)].drop_duplicates()
    if not duplicates.empty:
        duplicate_text = ", ".join(repr(value) for value in duplicates.iloc[:5])
        raise ValueError(
            f"{name} contains duplicate sequence_id values: {duplicate_text}"
        )
    out["sequence_id"] = sequence
    return out


def _validated_label_map(
    labels: dict[str, str] | None,
    *,
    name: str,
) -> dict[str, str] | None:
    """Normalize label-map sequence keys using the table sequence-ID boundary."""

    if labels is None:
        return None
    items = list(labels.items())
    if not items:
        return {}

    sequence = pd.Series([key for key, _ in items], dtype=object).astype("string").str.strip()
    missing_sequence = sequence.isna() | sequence.eq("")
    if bool(missing_sequence.any()):
        raise ValueError(f"{name} contains missing sequence_id keys")

    sequence = sequence.astype(str)
    duplicates = sequence.loc[sequence.duplicated(keep=False)].drop_duplicates()
    if not duplicates.empty:
        duplicate_text = ", ".join(repr(value) for value in duplicates.iloc[:5])
        raise ValueError(
            f"{name} contains duplicate sequence_id keys after normalization: {duplicate_text}"
        )
    return {
        sequence_id: str(value)
        for sequence_id, (_, value) in zip(sequence.tolist(), items, strict=True)
    }


def _sequence_indexed(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    """Index one per-sequence feature table without silently discarding rows."""

    out = pd.DataFrame(rows).copy()
    if out.empty:
        raise ValueError(f"{name} is empty")
    out = _validated_sequence_ids(out, name=name)
    return out.set_index("sequence_id", drop=True)


def _validated_probability_frame(rows: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Validate one per-sequence probability table before fusion."""

    out = pd.DataFrame(rows).copy()
    if "sequence_id" not in out.columns:
        if out.empty:
            out["sequence_id"] = pd.Series(dtype=str)
        else:
            raise ValueError(f"{name} must contain a sequence_id column")
    if out.empty:
        return out.reset_index(drop=True)

    out = _validated_sequence_ids(out, name=name)

    probability_columns = [
        f"predicted_probability_{label}"
        for label in _IMPL.OFFICIAL_SEQUENCE_CLASS_LABELS
        if f"predicted_probability_{label}" in out.columns
    ]
    if not probability_columns:
        raise ValueError(
            f"{name} must contain at least one official predicted_probability_* column"
        )

    invalid_rows = np.zeros(len(out), dtype=bool)
    numeric_columns: dict[str, pd.Series] = {}
    for column in probability_columns:
        raw = out[column]
        invalid_scalar = raw.map(
            lambda value: isinstance(
                value,
                (bool, np.bool_, complex, np.complexfloating),
            )
            or np.ma.is_masked(value)
        )
        clean = raw.astype(object).where(~invalid_scalar, np.nan)
        numeric = pd.to_numeric(clean, errors="coerce").astype(float)
        values = numeric.to_numpy(dtype=float)
        invalid_rows |= invalid_scalar.to_numpy(dtype=bool)
        invalid_rows |= ~np.isfinite(values) | (values < 0.0)
        numeric_columns[column] = numeric
    if bool(invalid_rows.any()):
        bad_sequences = out.loc[invalid_rows, "sequence_id"].astype(str).drop_duplicates()
        bad_text = ", ".join(repr(value) for value in bad_sequences.iloc[:5])
        raise ValueError(
            f"{name} must contain finite non-negative real probabilities; "
            f"invalid sequence_id values: {bad_text}"
        )

    out.loc[:, probability_columns] = pd.DataFrame(
        numeric_columns,
        index=out.index,
    )
    return out.reset_index(drop=True)


def _validate_fused_probability_mass(rows: pd.DataFrame) -> None:
    probability_columns = [
        f"predicted_probability_{label}"
        for label in _IMPL.OFFICIAL_SEQUENCE_CLASS_LABELS
    ]
    probability_mass = rows.loc[:, probability_columns].sum(axis=1).to_numpy(float)
    invalid = ~np.isfinite(probability_mass) | (probability_mass <= 0.0)
    if not bool(invalid.any()):
        return
    sequence_ids = rows.loc[invalid, "sequence_id"].astype(str).drop_duplicates()
    sequence_text = ", ".join(repr(value) for value in sequence_ids.iloc[:5])
    raise ValueError(
        "fused probabilities contain zero probability mass for sequence_id values: "
        f"{sequence_text}"
    )


class _FeatureCsvPandasProxy:
    """Delegate pandas while preserving opaque sequence ids on CSV reads."""

    def __init__(self, module: Any) -> None:
        self._module = module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def read_csv(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        options = dict(kwargs)
        if "dtype" not in options and "converters" not in options:
            options["dtype"] = {"sequence_id": "string"}
        return self._module.read_csv(*args, **options)


def main(argv: list[str] | None = None) -> int:
    """Run the legacy CLI while preserving opaque feature sequence identifiers."""

    original_pd = _IMPL.pd
    _IMPL.pd = _FeatureCsvPandasProxy(original_pd)
    try:
        return _LEGACY_MAIN(argv)
    finally:
        _IMPL.pd = original_pd


def fuse_sequence_probabilities(
    image_probabilities: pd.DataFrame,
    nonimage_probabilities: pd.DataFrame,
    *,
    image_weight: float,
    eval_labels: dict[str, str] | None = None,
    class_source: str | None = None,
) -> pd.DataFrame:
    """Blend validated image and non-image probability rows by sequence."""

    image_rows = _validated_probability_frame(
        image_probabilities,
        name="image_probabilities",
    )
    nonimage_rows = _validated_probability_frame(
        nonimage_probabilities,
        name="nonimage_probabilities",
    )
    if image_rows.empty and nonimage_rows.empty:
        raise ValueError("at least one probability source must contain rows")
    fused = _LEGACY_FUSE_SEQUENCE_PROBABILITIES(
        image_rows,
        nonimage_rows,
        image_weight=_validated_image_weight(image_weight),
        eval_labels=_validated_label_map(eval_labels, name="eval_labels"),
        class_source=class_source,
    )
    _validate_fused_probability_mass(fused)
    return fused


def select_train_safe_fusion(
    *,
    image_train_features: pd.DataFrame,
    nonimage_train_features: pd.DataFrame,
    image_predict_features: pd.DataFrame,
    nonimage_predict_features: pd.DataFrame,
    train_labels: dict[str, str],
    eval_labels: dict[str, str] | None = None,
    model_specs: list[FusionModelSpec],
    image_weights: list[float],
    cv_folds: int = 5,
    cv_random_state: int = 20260627,
    output_dir: Path | None = None,
) -> FusionSelectionResult:
    """Select fusion settings after validating every candidate weight."""

    validated_weights = [
        _validated_image_weight(value, name=f"image_weights[{index}]")
        for index, value in enumerate(image_weights)
    ]
    normalized_train_labels = _validated_label_map(train_labels, name="train_labels")
    normalized_eval_labels = _validated_label_map(eval_labels, name="eval_labels")
    if normalized_train_labels is None:  # pragma: no cover - signature excludes None
        raise ValueError("train_labels must be a label mapping")
    return _LEGACY_SELECT_TRAIN_SAFE_FUSION(
        image_train_features=image_train_features,
        nonimage_train_features=nonimage_train_features,
        image_predict_features=image_predict_features,
        nonimage_predict_features=nonimage_predict_features,
        train_labels=normalized_train_labels,
        eval_labels=normalized_eval_labels,
        model_specs=_distinct_model_specs(model_specs),
        image_weights=validated_weights,
        cv_folds=cv_folds,
        cv_random_state=cv_random_state,
        output_dir=output_dir,
    )


_IMPL._SelectionModelSpec = _SelectionModelSpec
_IMPL._distinct_model_specs = _distinct_model_specs
_IMPL._validated_image_weight = _validated_image_weight
_IMPL._validated_sequence_ids = _validated_sequence_ids
_IMPL._validated_label_map = _validated_label_map
_IMPL._sequence_indexed = _sequence_indexed
_IMPL._validated_probability_frame = _validated_probability_frame
_IMPL._validate_fused_probability_mass = _validate_fused_probability_mass
_IMPL.fuse_sequence_probabilities = fuse_sequence_probabilities
_IMPL.select_train_safe_fusion = select_train_safe_fusion
_IMPL.main = main

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_SelectionModelSpec"] = _SelectionModelSpec
globals()["_distinct_model_specs"] = _distinct_model_specs
globals()["_validated_image_weight"] = _validated_image_weight
globals()["_validated_sequence_ids"] = _validated_sequence_ids
globals()["_validated_label_map"] = _validated_label_map
globals()["_sequence_indexed"] = _sequence_indexed
globals()["_validated_probability_frame"] = _validated_probability_frame
globals()["_validate_fused_probability_mass"] = _validate_fused_probability_mass
globals()["fuse_sequence_probabilities"] = fuse_sequence_probabilities
globals()["select_train_safe_fusion"] = select_train_safe_fusion
globals()["main"] = main

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
