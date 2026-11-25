"""Regression tests for shared training command-line behavior."""

from __future__ import annotations

import argparse

import pytest
from forexmind.training.cli import (
    add_common_args,
    load_config,
    run_dir_for,
    run_multiseed,
)


def _args(*values: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_args(parser, algorithm="sac")
    return parser.parse_args(list(values))


def test_run_id_override_is_applied() -> None:
    config = load_config(_args("--run-id", "experiment"), "sac")

    assert config.run_id == "experiment"
    assert run_dir_for(config, 7).name == "sac_experiment_seed7"


def test_algorithm_prefix_is_not_duplicated_in_run_directory() -> None:
    config = load_config(_args("--config", "configs/sac_cpu.yaml"), "sac")

    assert config.run_id == "sac_cpu"
    assert run_dir_for(config, 42).name == "sac_cpu_seed42"


def test_launcher_rejects_mismatched_algorithm() -> None:
    with pytest.raises(ValueError, match="does not match"):
        run_multiseed(_args("--algorithm", "ppo"), "sac")
