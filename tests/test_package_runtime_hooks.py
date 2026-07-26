from __future__ import annotations

from collections.abc import Callable

import pytest

from raft_uav import _optional_runtime_hook


def _missing_module(name: str) -> Callable[[], Callable[[], None]]:
    def import_install() -> Callable[[], None]:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return import_install


@pytest.mark.parametrize("name", ["pyrecest", "pyrecest.filters"])
def test_optional_runtime_hook_ignores_missing_pyrecest_modules(name: str) -> None:
    _optional_runtime_hook(_missing_module(name))


@pytest.mark.parametrize("name", ["pyrecest_extra", "pyrecest_typo.filters"])
def test_optional_runtime_hook_reraises_similarly_named_missing_modules(
    name: str,
) -> None:
    with pytest.raises(ModuleNotFoundError, match=name):
        _optional_runtime_hook(_missing_module(name))
