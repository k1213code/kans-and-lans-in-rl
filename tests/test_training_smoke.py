from __future__ import annotations

import pandas as pd
import yaml

from configuration_layer.training import SB3PPOConfig
from execution_layer.runner import run_training


def make_smoke_config(
    tmp_path,
    *,
    run_id: str,
    use_custom_mlp_extractor: bool,
    backbone_type: str,
) -> SB3PPOConfig:
    return SB3PPOConfig(
        env_id="Pendulum-v1",
        seed=2025,
        device="cpu",
        cuda=False,
        total_timesteps=16,
        learning_rate=3e-4,
        use_lr_schedule=False,
        num_envs=1,
        num_steps=8,
        num_minibatches=2,
        update_epochs=1,
        use_custom_mlp_extractor=use_custom_mlp_extractor,
        actor_backbone_type=backbone_type,
        critic_backbone_type=backbone_type,
        actor_hidden_size=8,
        actor_num_hidden_layers=1,
        critic_hidden_size=8,
        critic_num_hidden_layers=1,
        grid_size=2,
        spline_order=1,
        enable_grid_updates=False,
        verbose=0,
        make_plots=False,
        track_memory=False,
        track_time=False,
        output_root=str(tmp_path),
        run_id=run_id,
    )


def assert_training_artifacts(model_path, *, backbone_type: str, expected_actor_label: str) -> None:
    run_dir = model_path.parent

    assert model_path.exists()
    assert run_dir.name.startswith("Pendulum-v1__")
    assert f"__actor-{expected_actor_label}__" in run_dir.name
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "progress.csv").exists()

    with (run_dir / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        resolved = yaml.safe_load(handle)

    assert resolved["env_id"] == "Pendulum-v1"
    assert resolved["actor_backbone_type"] == backbone_type
    assert resolved["critic_backbone_type"] == backbone_type

    progress = pd.read_csv(run_dir / "progress.csv")
    assert not progress.empty
    assert "time/total_timesteps" in progress.columns


def test_training_pipeline_smoke_writes_expected_artifacts(tmp_path) -> None:
    config = make_smoke_config(
        tmp_path,
        run_id="pytest-smoke-kan",
        use_custom_mlp_extractor=True,
        backbone_type="kan",
    )

    model_path = run_training(config)

    assert_training_artifacts(model_path, backbone_type="kan", expected_actor_label="kan")


def test_training_pipeline_smoke_covers_mlp_and_lan_backbones(tmp_path) -> None:
    for backbone_type in ["mlp", "lan"]:
        config = make_smoke_config(
            tmp_path,
            run_id=f"pytest-smoke-{backbone_type}",
            use_custom_mlp_extractor=True,
            backbone_type=backbone_type,
        )

        model_path = run_training(config)

        assert_training_artifacts(
            model_path,
            backbone_type=backbone_type,
            expected_actor_label=backbone_type,
        )


def test_training_pipeline_smoke_covers_standard_sb3_extractor(tmp_path) -> None:
    config = make_smoke_config(
        tmp_path,
        run_id="pytest-smoke-sb3",
        use_custom_mlp_extractor=False,
        backbone_type="mlp",
    )

    model_path = run_training(config)

    assert_training_artifacts(model_path, backbone_type="mlp", expected_actor_label="sb3")
