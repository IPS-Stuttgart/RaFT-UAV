"""Compatibility package guarding fixed-population CV output materialization."""

from __future__ import annotations

from functools import wraps
import importlib.util
from pathlib import Path
import sys
from typing import Any

from .._best_prediction_copy_guard import reject_best_prediction_copy_aliases

_IMPL_PATH = Path(__file__).resolve().parent.parent / "fixed_population_cv.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._fixed_population_cv_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load fixed-population CV implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_RUN_FIXED_POPULATION_CV = _IMPL.run_fixed_population_cv


@wraps(_ORIGINAL_RUN_FIXED_POPULATION_CV)
def run_fixed_population_cv(
    prediction_path: Path,
    truth_dir: Path,
    first_frame_label_dir: Path,
    output_dir: Path,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run cross-validation after ensuring best-output cleanup cannot delete an input."""

    reject_best_prediction_copy_aliases(
        Path(output_dir) / "best_predictions",
        prediction_path=Path(prediction_path),
        truth_dir=Path(truth_dir),
        first_frame_label_dir=Path(first_frame_label_dir),
    )
    return _ORIGINAL_RUN_FIXED_POPULATION_CV(
        Path(prediction_path),
        Path(truth_dir),
        Path(first_frame_label_dir),
        Path(output_dir),
        *args,
        **kwargs,
    )


_IMPL.run_fixed_population_cv = run_fixed_population_cv

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["run_fixed_population_cv"] = run_fixed_population_cv

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
