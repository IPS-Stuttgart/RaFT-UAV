"""Compatibility fixes for MMUAD layout token matching.

The maintained implementation lives in the sibling ``layout.py`` module. This
package preserves the public import path while requiring real token boundaries
for directory prefixes and short logical aliases such as ``gt``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys

_IMPL_PATH = Path(__file__).resolve().parent.parent / "layout.py"
_SPEC = importlib.util.spec_from_file_location(
    "raft_uav.mmuad._layout_legacy",
    _IMPL_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load layout implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _IMPL
_SPEC.loader.exec_module(_IMPL)


def _folded_prefix_has_boundary(name: str, token_folded: str) -> bool:
    """Return whether a folded token ends at a real name boundary."""

    consumed = 0
    raw_name = str(name)
    for index, character in enumerate(raw_name):
        if not character.isascii() or not character.isalnum():
            continue
        if consumed >= len(token_folded) or character.lower() != token_folded[consumed]:
            return False
        consumed += 1
        if consumed != len(token_folded):
            continue

        if index + 1 >= len(raw_name):
            return True
        next_character = raw_name[index + 1]
        if not next_character.isalnum():
            return True
        if next_character.isdigit():
            return True
        return character.islower() and next_character.isupper()
    return False


def _directory_name_matches_any(name: str, tokens: tuple[str, ...]) -> bool:
    """Match complete directory tokens without arbitrary word prefixes."""

    normalized = _IMPL._normalized_dir_name(name)
    folded = _IMPL._folded_dir_name(name)
    for token in tokens:
        token_normalized = _IMPL._normalized_dir_name(token)
        token_folded = _IMPL._folded_dir_name(token)
        if (
            normalized == token_normalized
            or normalized.startswith(f"{token_normalized}_")
            or folded == token_folded
            or (
                folded.startswith(token_folded)
                and _folded_prefix_has_boundary(name, token_folded)
            )
        ):
            return True
    return False


def _logical_text_has_any(parts: tuple[str, ...], tokens: tuple[str, ...]) -> bool:
    """Match logical file tokens without letting short aliases hit word interiors."""

    normalized = " ".join(_IMPL._normalized_dir_name(part) for part in parts)
    folded = _IMPL._folded_dir_name(normalized)
    lexical_tokens = {
        item for item in re.split(r"[^a-z0-9]+", normalized) if item
    }
    for token in tokens:
        token_normalized = _IMPL._normalized_dir_name(token)
        token_folded = _IMPL._folded_dir_name(token)
        if len(token_folded) <= 2:
            if token_folded in lexical_tokens:
                return True
            continue
        if token_normalized in normalized or token_folded in folded:
            return True
    return False


_IMPL._folded_prefix_has_boundary = _folded_prefix_has_boundary
_IMPL._directory_name_matches_any = _directory_name_matches_any
_IMPL._logical_text_has_any = _logical_text_has_any

globals().update(
    {
        name: getattr(_IMPL, name)
        for name in dir(_IMPL)
        if not (name.startswith("__") and name.endswith("__"))
    }
)
globals()["_folded_prefix_has_boundary"] = _folded_prefix_has_boundary
globals()["_directory_name_matches_any"] = _directory_name_matches_any
globals()["_logical_text_has_any"] = _logical_text_has_any

__doc__ = _IMPL.__doc__
__all__ = [
    name for name in dir(_IMPL) if not (name.startswith("__") and name.endswith("__"))
]
