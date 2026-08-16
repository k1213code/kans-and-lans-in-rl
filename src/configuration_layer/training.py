"""Parse YAML + CLI values into one training config."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from typing import Any

from utility_layer.cli_parsing import str2bool
from utility_layer.config_io import load_config_file
from utility_layer.paths import DEFAULT_OUTPUT_ROOT_TEXT, resolve_config_path


@dataclass
class SB3PPOConfig:
    """Resolved training configuration used everywhere else in the project."""

    # Experiment label.
    exp_name: str = "sb3_ppo_kan_mlp"
    # Random seed for envs, NumPy, Python, and PyTorch.
    seed: int = 1
    # Whether GPU usage is allowed when device="auto".
    cuda: bool = False
    # Device choice: "auto", "cpu", or "cuda".
    device: str = "auto"
    # Gymnasium environment id to train on.
    env_id: str = "Walker2d-v5"
    # Total number of environment steps to train for.
    total_timesteps: int = 10_000_000
    # Initial learning rate for PPO.
    learning_rate: float = 3e-4
    # Whether to use the project's custom LR schedule.
    use_lr_schedule: bool = False
    # Number of steps to keep the initial LR unchanged.
    lr_hold_steps: int = 2_000_000
    # Learning rate after the decay phase finishes.
    final_learning_rate: float = 5e-5
    # Number of steps over which LR decays linearly.
    lr_decay_steps: int = 2_000_000
    # Number of parallel environments.
    num_envs: int = 4
    # Rollout length collected per environment before one PPO update.
    num_steps: int = 2048 # Rollout size = num_envs * num_steps = 8192 Samples | updated PPO every 8192 Steps
    # Number of PPO minibatches per rollout batch.
    num_minibatches: int = 32 # minibatch_size = 8192/32 = 256
    # Number of gradient passes over each rollout batch.
    update_epochs: int = 10 #optimizer steps per PPO-Update = minibatches * epochs = 320
    # Discount factor for future rewards.
    gamma: float = 0.99
    # GAE smoothing parameter for advantage estimation.
    gae_lambda: float = 0.95
    # PPO policy clipping range.
    clip_coef: float = 0.2
    # Entropy bonus weight for exploration.
    ent_coef: float = 0.0
    # Value loss weight.
    vf_coef: float = 0.5
    # Gradient clipping threshold.
    max_grad_norm: float = 0.5
    # Optional early-stop threshold for KL divergence.
    target_kl: float | None = None
    # Whether to replace SB3's actor/critic mlp_extractor with custom backbones.
    use_custom_mlp_extractor: bool = False
    # Backbone type used in the actor branch.
    actor_backbone_type: str = "mlp"
    # Backbone type used in the critic branch.
    critic_backbone_type: str = "mlp"
    # Hidden width of the actor backbone.
    actor_hidden_size: int = 64
    # Number of hidden layers in the actor backbone.
    actor_num_hidden_layers: int = 2
    # Hidden width of the critic backbone.
    critic_hidden_size: int = 64
    # Number of hidden layers in the critic backbone.
    critic_num_hidden_layers: int = 2
    # Number of spline intervals in KAN/LAN grids.
    grid_size: int = 3
    # B-spline order used by spline-based backbones.
    spline_order: int = 2
    # Blend factor between adaptive and uniform grid updates.
    grid_eps: float = 0.02
    # Lower bound of the initial spline grid.
    grid_min: float = -1.0
    # Upper bound of the initial spline grid.
    grid_max: float = 1.0
    # SB3 verbosity level.
    verbose: int = 1
    # Optional TensorBoard log directory.
    tensorboard_log: str = ""
    # Optional final model output path.
    save_path: str = ""
    # Save a checkpoint every N callback calls; 0 disables checkpoints.
    checkpoint_freq: int = 0
    # Whether to create plots after training.
    make_plots: bool = True
    # Smoothing window used in generated training plots.
    smooth_window: int = 20
    # Noise scale used when initializing spline coefficients.
    scale_noise: float = 0.1
    # Whether LAN/KAN-style modules keep a standard base activation branch.
    use_base_branch: bool = True
    # Reserved option for periodic branches in future backbone variants.
    use_periodic_branch: bool = True
    # Reserved maximum periodic frequency.
    max_freq: float = 3.0
    # Reserved initial periodic frequency.
    init_freq: float = 1.0
    # Reserved scale for periodic branch contribution.
    scale_periodic: float = 0.05
    # Whether spline grids may adapt during training.
    enable_grid_updates: bool = False
    # Try a grid update every N forward passes.
    grid_update_every: int = 100
    # Stop attempting grid updates after this many forward passes.
    grid_update_until: int = 30000
    # Whether to log memory statistics during training.
    track_memory: bool = True
    # Log memory metrics every N callback calls.
    memory_log_every: int = 100
    # Whether to log timing statistics during training.
    track_time: bool = True
    # Log timing metrics every N callback calls.
    time_log_every: int = 100
    # Root directory for all run outputs.
    output_root: str = DEFAULT_OUTPUT_ROOT_TEXT
    # Optional externally provided run id; empty means auto-generate one.
    run_id: str = ""


def _config_field_names() -> set[str]:
    """Return all configuration keys accepted by SB3PPOConfig.

    The dataclass defines valid arguments. If a field is added or removed
    there, YAML validation automatically follows that change.
    """
    return {field.name for field in fields(SB3PPOConfig)}


def _filter_config_values(values: dict[str, Any]) -> dict[str, Any]:
    """Validate values loaded from YAML before they are merged into the config."""
    valid_fields = _config_field_names()
    unknown = sorted(set(values) - valid_fields)
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(unknown)}")

    # Keep only real training config fields.
    return {key: value for key, value in values.items() if key in valid_fields}


def build_train_parser() -> argparse.ArgumentParser:
    # Keeping defaults at None makes the later merge with YAML easier.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Optional YAML config file.")
    parser.add_argument("--exp_name", type=str, default=None, help="Human-readable experiment label.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--cuda", type=str2bool, default=None, help="Allow CUDA when device is auto.")
    parser.add_argument("--device", type=str, default=None, help="Device to use: auto, cpu, or cuda.")
    parser.add_argument("--env_id", type=str, default=None, help="Gymnasium environment id.")
    parser.add_argument("--total_timesteps", type=int, default=None, help="Total number of environment steps.")
    parser.add_argument("--learning_rate", type=float, default=None, help="Initial PPO learning rate.")
    parser.add_argument("--use_lr_schedule", type=str2bool, default=None, help="Enable hold-then-decay LR schedule.")
    parser.add_argument("--lr_hold_steps", type=int, default=None, help="Steps to keep the initial learning rate.")
    parser.add_argument("--lr_decay_steps", type=int, default=None, help="Steps over which LR decays linearly.")
    parser.add_argument("--final_learning_rate", type=float, default=None, help="Learning rate after decay completes.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
    parser.add_argument("--num_steps", type=int, default=None, help="Rollout steps collected per environment.")
    parser.add_argument("--num_minibatches", type=int, default=None, help="Number of PPO minibatches per rollout.")
    parser.add_argument("--update_epochs", type=int, default=None, help="Gradient epochs per rollout batch.")
    parser.add_argument("--gamma", type=float, default=None, help="Reward discount factor.")
    parser.add_argument("--gae_lambda", type=float, default=None, help="GAE lambda parameter.")
    parser.add_argument("--clip_coef", type=float, default=None, help="PPO clipping range.")
    parser.add_argument("--ent_coef", type=float, default=None, help="Entropy bonus weight.")
    parser.add_argument("--vf_coef", type=float, default=None, help="Value loss weight.")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Gradient clipping threshold.")
    parser.add_argument("--target_kl", type=float, default=None, help="Optional KL threshold for early stopping.")
    parser.add_argument("--grid_size", type=int, default=None, help="Number of spline intervals in KAN/LAN.")
    parser.add_argument("--spline_order", type=int, default=None, help="Order of the B-spline basis.")
    parser.add_argument("--grid_eps", type=float, default=None, help="Blend factor between adaptive and uniform grid.")
    parser.add_argument("--grid_min", type=float, default=None, help="Lower bound of the initial spline grid.")
    parser.add_argument("--grid_max", type=float, default=None, help="Upper bound of the initial spline grid.")
    parser.add_argument("--verbose", type=int, default=None, help="SB3 verbosity level.")
    parser.add_argument("--tensorboard_log", type=str, default=None, help="TensorBoard log directory.")
    parser.add_argument("--save_path", type=str, default=None, help="Path for the final saved model.")
    parser.add_argument("--checkpoint_freq", type=int, default=None, help="Save checkpoints every N callback calls.")
    parser.add_argument("--make_plots", type=str2bool, default=None, help="Create plots after training finishes.")
    parser.add_argument("--smooth_window", type=int, default=None, help="Smoothing window used in generated plots.")
    parser.add_argument("--enable_grid_updates", type=str2bool, default=None, help="Allow spline grids to adapt during training.")
    parser.add_argument("--grid_update_every", type=int, default=None, help="Attempt grid updates every N forward passes.")
    parser.add_argument("--grid_update_until", type=int, default=None, help="Stop grid updates after this many forward passes.")
    parser.add_argument("--track_memory", type=str2bool, default=None, help="Log memory usage during training.")
    parser.add_argument("--memory_log_every", type=int, default=None, help="Memory logging interval in callback calls.")
    parser.add_argument("--track_time", type=str2bool, default=None, help="Log timing information during training.")
    parser.add_argument("--time_log_every", type=int, default=None, help="Timing logging interval in callback calls.")
    parser.add_argument("--use_custom_mlp_extractor", type=str2bool, default=None, help="Replace SB3 actor/critic mlp_extractor with custom backbones.")
    parser.add_argument("--actor_backbone_type", type=str, choices=["mlp", "kan", "kan_no_base", "kan_no_spline", "lan", "debug_constant"], default=None, help="Backbone type for the actor branch.",)
    parser.add_argument("--critic_backbone_type", type=str, choices=["mlp", "kan", "kan_no_base", "kan_no_spline", "lan", "debug_constant"], default=None, help="Backbone type for the critic branch.",)
    parser.add_argument("--actor_hidden_size", type=int, default=None, help="Hidden width of the actor backbone.")
    parser.add_argument("--actor_num_hidden_layers", type=int, default=None, help="Number of hidden layers in the actor backbone.")
    parser.add_argument("--critic_hidden_size", type=int, default=None, help="Hidden width of the critic backbone.")
    parser.add_argument("--critic_num_hidden_layers", type=int, default=None, help="Number of hidden layers in the critic backbone.")
    parser.add_argument("--scale_noise", type=float, default=None, help="Initialization noise scale for spline coefficients.")
    parser.add_argument("--use_base_branch", type=str2bool, default=None, help="Keep the standard base activation branch in spline modules.")
    parser.add_argument("--use_periodic_branch", type=str2bool, default=None, help="Reserved flag for periodic branches in future variants.")
    parser.add_argument("--max_freq", type=float, default=None, help="Reserved maximum periodic frequency.")
    parser.add_argument("--init_freq", type=float, default=None, help="Reserved initial periodic frequency.")
    parser.add_argument("--scale_periodic", type=float, default=None, help="Reserved periodic branch scale.")
    parser.add_argument("--output_root", type=str, default=None, help="Root directory for run outputs.")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run id; auto-generated if omitted.")
    return parser


def parse_train_config(argv: list[str] | None = None) -> SB3PPOConfig:
    """Build the final training config from defaults, optional YAML, and CLI overrides.

    Precedence is:
    code defaults < YAML config file < command-line arguments.
    """

    # Parse the CLI first because --config tells us which YAML file to load.
    parser = build_train_parser()
    namespace = parser.parse_args(argv)
    cli_values = {
        key: value
        for key, value in vars(namespace).items()
        # config only selects the YAML file. It is not a field of SB3PPOConfig.
        if value is not None and key != "config"
    }

    # YAML is optional, so command-line-only runs and default-only runs still work.
    file_values: dict[str, Any] = {}
    if namespace.config:
        file_values = _filter_config_values(load_config_file(resolve_config_path(namespace.config)))

    # Start with dataclass defaults, then overlay YAML values, then CLI values.
    merged_values = asdict(SB3PPOConfig())
    merged_values.update(file_values)
    merged_values.update(cli_values)
    return SB3PPOConfig(**merged_values)
