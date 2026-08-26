from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_nested_lofo_tuning.py"
spec = importlib.util.spec_from_file_location("run_nested_lofo_tuning_script", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_candidate_options_reject_string_instead_of_splitting_characters() -> None:
    with pytest.raises(ValueError, match="candidate options must be a list"):
        module._format_candidate_values("--smoother fixed-lag", {})


def test_candidate_options_format_list_values() -> None:
    assert module._format_candidate_values(
        ["--output-dir", "{output_dir}", "--flag"],
        {"output_dir": "output root"},
    ) == ["--output-dir", "output root", "--flag"]


def test_load_candidates_rejects_non_object_entries(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text('["not-an-object"]', encoding="utf-8")

    with pytest.raises(ValueError, match="candidate entry 0 must be an object"):
        module._load_candidates(path)
