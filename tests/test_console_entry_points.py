from pathlib import Path
import tomllib


def _project_scripts() -> dict[str, str]:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["scripts"]


def test_range_tracklet_viterbi_cli_is_installed_as_console_script():
    scripts = _project_scripts()

    assert (
        scripts["raft-uav-tracklet-viterbi-range"]
        == "raft_uav.tracklet_viterbi_range_cli:main"
    )
