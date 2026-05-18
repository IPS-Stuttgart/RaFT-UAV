from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_tracklet_viterbi_ablation.py"


def _load_ablation_module():
    spec = importlib.util.spec_from_file_location("run_tracklet_viterbi_ablation", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_staged_tracklet_variants_map_to_expected_runners_and_catprob_modes() -> None:
    module = _load_ablation_module()

    expected = {
        "base": ("base", "hard"),
        "retention-hard": ("retention", "hard"),
        "soft-catprob": ("retention", "soft"),
        "support": ("retention", "soft"),
        "range-covariance": ("range-covariance", "soft"),
    }

    for variant, (runner_variant, catprob_mode) in expected.items():
        assert module._runner_variant(variant) == runner_variant
        assert module._catprob_retention_mode(variant) == catprob_mode


def test_staged_tracklet_environment_contains_all_wrapper_knobs() -> None:
    module = _load_ablation_module()
    config = module._Config(
        name="dummy",
        threshold=0.4,
        variant="support",
        runner_variant="retention",
        catprob_retention_mode="soft",
        max_candidates_per_frame=9,
        max_candidate_pool_per_frame=18,
        max_candidates_per_track_id=2,
        below_catprob_threshold_penalty=2.5,
        track_support_weight=0.45,
        max_track_support_reward=4.0,
    )

    env = module._tracklet_environment(config)

    assert env[module._TRACKLET_VARIANT_ENV] == "retention"
    assert env[module._CATPROB_MODE_ENV] == "soft"
    assert env[module._MAX_CANDIDATES_PER_FRAME_ENV] == "9"
    assert env[module._MAX_CANDIDATE_POOL_ENV] == "18"
    assert env[module._MAX_CANDIDATES_PER_TRACK_ENV] == "2"
    assert env[module._BELOW_CATPROB_PENALTY_ENV] == "2.5"
    assert env[module._TRACK_SUPPORT_WEIGHT_ENV] == "0.45"
    assert env[module._MAX_TRACK_SUPPORT_REWARD_ENV] == "4.0"
