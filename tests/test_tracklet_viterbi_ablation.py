from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_tracklet_viterbi_ablation as ablation  # noqa: E402


def _args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "thresholds": [0.4],
        "tracklet_max_candidates_per_frame": [8],
        "tracklet_max_candidate_pool_per_frame": [16],
        "tracklet_max_candidates_per_track_id": [1],
        "tracklet_below_catprob_threshold_penalties": [3.0],
        "tracklet_track_support_weights": [0.0, 0.45],
        "tracklet_max_track_support_rewards": [4.0],
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_configs_build_controlled_support_ablation_grid() -> None:
    configs = ablation._configs(_args())

    assert len(configs) == 2
    assert [config.track_support_weight for config in configs] == [0.0, 0.45]
    assert [config.max_track_support_reward for config in configs] == [4.0, 4.0]
    assert configs[0].name == "tracklet_t0p40_k8_pool16_perid1_catpen3p0_sw0p00_sr4p0"
    assert configs[1].name == "tracklet_t0p40_k8_pool16_perid1_catpen3p0_sw0p45_sr4p0"


def test_configs_cross_product_includes_support_reward_sweep() -> None:
    configs = ablation._configs(
        _args(
            thresholds=[0.3, 0.4],
            tracklet_track_support_weights=[0.0, 0.25],
            tracklet_max_track_support_rewards=[2.0, 4.0],
        )
    )

    assert len(configs) == 8
    assert {config.threshold for config in configs} == {0.3, 0.4}
    assert {config.track_support_weight for config in configs} == {0.0, 0.25}
    assert {config.max_track_support_reward for config in configs} == {2.0, 4.0}


def test_tracklet_environment_maps_config_to_wrapper_variables() -> None:
    config = ablation._configs(_args(tracklet_track_support_weights=[0.45]))[0]

    environment = ablation._tracklet_environment(config)

    assert environment[ablation._MAX_CANDIDATE_POOL_ENV] == "16"
    assert environment[ablation._MAX_CANDIDATES_PER_TRACK_ENV] == "1"
    assert environment[ablation._BELOW_CATPROB_PENALTY_ENV] == "3.0"
    assert environment[ablation._TRACK_SUPPORT_WEIGHT_ENV] == "0.45"
    assert environment[ablation._MAX_TRACK_SUPPORT_REWARD_ENV] == "4.0"


def test_environment_overrides_are_restored() -> None:
    key = ablation._TRACK_SUPPORT_WEIGHT_ENV
    old_value = os.environ.get(key)
    previous = ablation._set_environment({key: "0.0"})
    try:
        assert os.environ[key] == "0.0"
    finally:
        ablation._restore_environment(previous)

    assert os.environ.get(key) == old_value


def test_validate_args_rejects_empty_support_weight_sweep() -> None:
    try:
        ablation._validate_args(_args(tracklet_track_support_weights=[]))
    except SystemExit as exc:
        assert "--tracklet-track-support-weights" in str(exc)
    else:
        raise AssertionError("expected SystemExit for empty support-weight sweep")
