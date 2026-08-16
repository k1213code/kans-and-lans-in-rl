from __future__ import annotations

from pathlib import Path

import pytest

from configuration_layer.training import SB3PPOConfig, parse_train_config
from execution_layer.builders import build_ppo_kwargs
from utility_layer.paths import CONFIG_ROOT


CONFIG_PATHS = sorted(CONFIG_ROOT.rglob("*.yaml"))
BACKBONE_TYPES = {"mlp", "kan", "kan_no_base", "kan_no_spline", "lan", "debug_constant"}


def config_id(path: Path) -> str:
    return path.relative_to(CONFIG_ROOT).as_posix()


def test_parse_train_config_merges_yaml_and_cli_overrides() -> None:
    config = parse_train_config(
        [
            "--config",
            "general_experiments/walker2d_kan.yaml",
            "--seed",
            "123",
            "--device",
            "cpu",
            "--total_timesteps",
            "32",
            "--enable_grid_updates",
            "false",
        ]
    )

    assert config.env_id == "Walker2d-v5"
    assert config.actor_backbone_type == "kan"
    assert config.critic_backbone_type == "kan"
    assert config.seed == 123
    assert config.device == "cpu"
    assert config.total_timesteps == 32
    assert config.enable_grid_updates is False


@pytest.mark.parametrize("config_path", CONFIG_PATHS, ids=config_id)
def test_all_yaml_training_configs_parse(config_path: Path) -> None:
    config = parse_train_config(["--config", str(config_path)])

    assert config.env_id
    assert config.actor_backbone_type in BACKBONE_TYPES
    assert config.critic_backbone_type in BACKBONE_TYPES
    assert config.total_timesteps > 0
    assert config.num_envs > 0
    assert config.num_steps > 0
    assert config.num_minibatches > 0


def test_config_paths_can_use_configs_prefix() -> None:
    config = parse_train_config(["--config", "configs/general_experiments/walker2d_mlp.yaml"])

    assert config.env_id == "Walker2d-v5"
    assert config.actor_backbone_type == "mlp"


def test_parse_train_config_rejects_unknown_yaml_keys(tmp_path) -> None:
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text("env_id: Pendulum-v1\nunknown_field: true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown config fields: unknown_field"):
        parse_train_config(["--config", str(config_path)])


def test_build_ppo_kwargs_requires_divisible_minibatches() -> None:
    config = SB3PPOConfig(
        num_envs=3,
        num_steps=5,
        num_minibatches=4,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        build_ppo_kwargs(config)


def test_build_ppo_kwargs_maps_config_to_policy_kwargs() -> None:
    config = SB3PPOConfig(
        use_custom_mlp_extractor=True,
        actor_backbone_type="lan",
        critic_backbone_type="kan",
        actor_hidden_size=16,
        critic_hidden_size=32,
        grid_size=5,
        spline_order=3,
        grid_min=-2.0,
        grid_max=2.0,
        num_envs=2,
        num_steps=8,
        num_minibatches=4,
        use_lr_schedule=False,
    )

    kwargs = build_ppo_kwargs(config)
    policy_kwargs = kwargs["policy_kwargs"]

    assert kwargs["batch_size"] == 4
    assert kwargs["learning_rate"] == config.learning_rate
    assert policy_kwargs["net_arch"] == []
    assert policy_kwargs["actor_backbone_type"] == "lan"
    assert policy_kwargs["critic_backbone_type"] == "kan"
    assert policy_kwargs["actor_hidden_size"] == 16
    assert policy_kwargs["critic_hidden_size"] == 32
    assert policy_kwargs["kan_kwargs"]["grid_size"] == 5
    assert policy_kwargs["kan_kwargs"]["spline_order"] == 3
    assert policy_kwargs["kan_kwargs"]["grid_range"] == [-2.0, 2.0]
