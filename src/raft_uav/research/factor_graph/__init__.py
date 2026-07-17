"""Compatibility package for the factor-graph research utilities."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_LEGACY_PATH = Path(__file__).resolve().parent.parent / "factor_graph.py"
_LEGACY_NAME = f"{__name__.rsplit('.', 1)[0]}._factor_graph_legacy"
_SPEC = importlib.util.spec_from_file_location(_LEGACY_NAME, _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise ImportError(f"cannot load factor-graph implementation from {_LEGACY_PATH}")
_LEGACY = importlib.util.module_from_spec(_SPEC)
sys.modules[_LEGACY_NAME] = _LEGACY
_SPEC.loader.exec_module(_LEGACY)

for _name in dir(_LEGACY):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_LEGACY, _name)
