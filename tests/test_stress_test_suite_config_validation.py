from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_stress_test_suite import _load_configs, main


def _write_configs(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "stress_configs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "name",
    ["../escape", "nested/run", "nested\\run", ".", "..", "", "   ", 17],
)
def test_load_configs_rejects_unsafe_output_names(tmp_path: Path, name: object) -> None:
    path = _write_configs(tmp_path, [{"name": name}])

    with pytest.raises(ValueError, match="stress config at index 0 name"):
        _load_configs(path)


def test_main_rejects_unsafe_name_before_creating_output_dir(tmp_path: Path) -> None:
    radar_path = tmp_path / "radar.csv"
    radar_path.write_text(
        "time_s,east_m,north_m,up_m\n0,0,0,0\n",
        encoding="utf-8",
    )
    config_path = _write_configs(tmp_path, [{"name": "../escaped"}])
    output_dir = tmp_path / "suite"

    with pytest.raises(ValueError, match="single safe directory name"):
        main(
            [
                "--radar-csv",
                str(radar_path),
                "--configs-json",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert not output_dir.exists()
    assert not (tmp_path / "escaped").exists()


def test_load_configs_rejects_duplicate_output_names(tmp_path: Path) -> None:
    path = _write_configs(
        tmp_path,
        [
            {"name": "same", "radar_drop_rate": 0.1},
            {"name": "same", "radar_drop_rate": 0.9},
        ],
    )

    with pytest.raises(ValueError, match="duplicate stress config name: 'same'"):
        _load_configs(path)


def test_load_configs_rejects_unknown_fields(tmp_path: Path) -> None:
    path = _write_configs(
        tmp_path,
        [{"name": "typo", "radar_drop_raet": 0.5}],
    )

    with pytest.raises(ValueError, match="unknown field.*radar_drop_raet"):
        _load_configs(path)


def test_load_configs_rejects_non_object_entries(tmp_path: Path) -> None:
    path = _write_configs(tmp_path, ["not-an-object"])

    with pytest.raises(ValueError, match="index 0 must be a JSON object"):
        _load_configs(path)


def test_load_configs_rejects_missing_name(tmp_path: Path) -> None:
    path = _write_configs(tmp_path, [{"radar_drop_rate": 0.5}])

    with pytest.raises(ValueError, match="index 0 must define 'name'"):
        _load_configs(path)


def test_load_configs_preserves_valid_numeric_string_controls(tmp_path: Path) -> None:
    path = _write_configs(
        tmp_path,
        [
            {
                "name": "serialized",
                "radar_drop_rate": "0.25",
                "false_tracks_per_frame": "2",
                "seed": "17",
            }
        ],
    )

    [config] = _load_configs(path)
    assert config.name == "serialized"
    assert config.radar_drop_rate == pytest.approx(0.25)
    assert config.false_tracks_per_frame == 2
    assert config.seed == 17
