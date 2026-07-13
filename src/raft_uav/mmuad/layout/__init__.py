"""Package wrapper that hardens archived MMUAD topic-map inspection.

The legacy implementation lives in the sibling ``layout.py`` file. This wrapper
preserves public imports while treating malformed archived topic-map bytes as
ordinary metadata instead of aborting the entire layout inventory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from raft_uav.mmuad.layout_archive_text_guard import patch_module as _patch_layout_module

_IMPL_PATH = Path(__file__).resolve().parent.parent / "layout.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._layout_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load legacy MMUAD layout helpers from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)

_patch_layout_module(_IMPL)

for _name in dir(_IMPL):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_IMPL, _name)

__doc__ = _IMPL.__doc__
__all__ = [_name for _name in dir(_IMPL) if not _name.startswith("__")]
