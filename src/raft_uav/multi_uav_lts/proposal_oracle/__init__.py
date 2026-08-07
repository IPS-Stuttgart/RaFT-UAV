"""Compatibility guard for Multi-UAV LTS proposal-oracle output paths.

The maintained implementation lives in the sibling ``proposal_oracle.py``
module. This package preserves the public import path while rejecting derived
oracle-output directories that alias truth or proposal inputs before stale
output cleanup can delete those inputs.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

_IMPL_PATH = Path(__file__).resolve().parent.parent / "proposal_oracle.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.multi_uav_lts._proposal_oracle_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load proposal-oracle implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_ORIGINAL_AUDIT_PROPOSAL_BANKS = _IMPL.audit_proposal_banks


def _reject_derived_oracle_output_aliases(
    proposal_paths: Mapping[str, Path],
    truth_dir: Path,
    output_dir: Path,
    *,
    include_fused: bool,
) -> None:
    """Reject cleanup targets that resolve to an input directory."""

    normalized_paths = _IMPL._normalize_proposal_paths(proposal_paths)
    source_names = list(normalized_paths)
    if include_fused and len(source_names) > 1:
        source_names.append("fused")

    oracle_root = Path(output_dir).expanduser().resolve() / "oracle_predictions"
    oracle_dirs = {
        (oracle_root / source).resolve(): source
        for source in source_names
    }
    truth = Path(truth_dir).expanduser().resolve()
    for oracle_dir, source in oracle_dirs.items():
        if oracle_dir == truth:
            raise ValueError(
                f"oracle output directory for source {source!r} "
                "must not alias the truth directory"
            )

    for input_name, input_path in normalized_paths.items():
        proposal = Path(input_path).expanduser().resolve()
        for oracle_dir, source in oracle_dirs.items():
            if oracle_dir == proposal:
                raise ValueError(
                    f"oracle output directory for source {source!r} "
                    f"must not alias proposal source {input_name!r}"
                )


def audit_proposal_banks(
    proposal_paths: Mapping[str, Path],
    truth_dir: Path,
    output_dir: Path,
    *,
    confidence_thresholds: Sequence[float] = _IMPL.DEFAULT_CONFIDENCE_THRESHOLDS,
    iou_thresholds: Sequence[float] = _IMPL.DEFAULT_IOU_THRESHOLDS,
    oracle_confidence_threshold: float = 0.003,
    oracle_iou_threshold: float = 0.05,
    include_fused: bool = True,
    sequences: Iterable[str] | None = None,
) -> Any:
    """Audit proposal banks after validating every derived cleanup target."""

    _reject_derived_oracle_output_aliases(
        proposal_paths,
        truth_dir,
        output_dir,
        include_fused=include_fused,
    )
    return _ORIGINAL_AUDIT_PROPOSAL_BANKS(
        proposal_paths,
        truth_dir,
        output_dir,
        confidence_thresholds=confidence_thresholds,
        iou_thresholds=iou_thresholds,
        oracle_confidence_threshold=oracle_confidence_threshold,
        oracle_iou_threshold=oracle_iou_threshold,
        include_fused=include_fused,
        sequences=sequences,
    )


_IMPL._reject_derived_oracle_output_aliases = _reject_derived_oracle_output_aliases
_IMPL.audit_proposal_banks = audit_proposal_banks

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_reject_derived_oracle_output_aliases"] = (
    _reject_derived_oracle_output_aliases
)
globals()["audit_proposal_banks"] = audit_proposal_banks

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
