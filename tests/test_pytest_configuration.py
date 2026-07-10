from __future__ import annotations

import configparser
from pathlib import Path


def test_repository_pytest_configuration_keeps_isolated_tempdir_and_markers() -> None:
    parser = configparser.ConfigParser()
    config_path = Path("pytest.ini")

    assert config_path.exists()
    assert parser.read(config_path, encoding="utf-8") == [str(config_path)]

    pytest_config = parser["pytest"]
    assert pytest_config.get("testpaths", "").split() == ["tests"]
    assert pytest_config.get("pythonpath", "").split() == ["src", "."]
    assert "--basetemp=.pytest-tmp" in pytest_config.get("addopts", "").split()

    marker_names = {
        line.partition(":")[0].strip()
        for line in pytest_config.get("markers", "").splitlines()
        if line.strip()
    }
    assert "integration" in marker_names
