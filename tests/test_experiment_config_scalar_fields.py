from raft_uav.experiments.config import ExperimentConfig


def test_from_mapping_treats_scalar_sequence_fields_as_single_values() -> None:
    config = ExperimentConfig.from_mapping(
        {
            "flights": "Opt1",
            "methods": "kalman",
            "options": "rerun",
        }
    )

    assert config.flights == ("Opt1",)
    assert config.methods == ("kalman",)
    assert config.options == ("rerun",)


def test_from_mapping_preserves_empty_and_iterable_sequence_fields() -> None:
    config = ExperimentConfig.from_mapping(
        {
            "flights": "",
            "methods": ["kalman", "imm"],
            "options": ("original", "gated"),
        }
    )

    assert config.flights == ()
    assert config.methods == ("kalman", "imm")
    assert config.options == ("original", "gated")
