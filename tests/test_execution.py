from __future__ import annotations

from configuration_layer.training import SB3PPOConfig
from execution_layer.env_factory import build_run_name, resolve_device, resolve_run_dir
from execution_layer.run_naming import compose_run_name


def test_compose_run_name_uses_backbones_for_custom_extractor() -> None:
    run_name = compose_run_name(
        env_id="Pendulum-v1",
        seed=7,
        use_custom_mlp_extractor=True,
        actor_backbone_type="kan",
        critic_backbone_type="lan",
        run_id="abc",
    )

    assert run_name == "Pendulum-v1__actor-kan__critic-lan__seed7__run-abc"


def test_compose_run_name_uses_sb3_label_for_standard_extractor() -> None:
    run_name = compose_run_name(
        env_id="Pendulum-v1",
        seed=7,
        use_custom_mlp_extractor=False,
        actor_backbone_type="kan",
        critic_backbone_type="lan",
        run_id="abc",
    )

    assert run_name == "Pendulum-v1__actor-sb3__critic-sb3__seed7__run-abc"


def test_resolve_run_dir_uses_output_root_and_runs_subfolder(tmp_path) -> None:
    config = SB3PPOConfig(
        env_id="Pendulum-v1",
        output_root=str(tmp_path),
        run_id="fixed",
        use_custom_mlp_extractor=True,
        actor_backbone_type="mlp",
        critic_backbone_type="mlp",
    )

    run_name = build_run_name(config)
    run_dir = resolve_run_dir(config, run_name)

    assert run_dir == tmp_path / "runs" / "Pendulum-v1__actor-mlp__critic-mlp__seed1__run-fixed"


def test_resolve_device_respects_explicit_cpu() -> None:
    config = SB3PPOConfig(device="cpu", cuda=True)

    assert resolve_device(config) == "cpu"
