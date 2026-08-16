"""Translate the project config into the kwargs that SB3 expects."""

from __future__ import annotations

from typing import Any
import torch.nn as nn


def build_policy_kwargs(args) -> dict[str, Any]:
    """
    Build the kwargs that will be passed into the SB3 policy class.
    """
    return {
        "net_arch": [] if args.use_custom_mlp_extractor else dict(pi=[64, 64], vf=[64, 64]),

        # FlexibleActorCriticPolicy branches on this flag in _build_mlp_extractor().
        "use_custom_mlp_extractor": args.use_custom_mlp_extractor,

        # Actor and critic stay separate because mixed setups are potentially useful.
        "actor_backbone_type": args.actor_backbone_type,
        "critic_backbone_type": args.critic_backbone_type,

        "actor_hidden_size": args.actor_hidden_size,
        "actor_num_hidden_layers": args.actor_num_hidden_layers,
        "actor_activation_fn": nn.SiLU, # Activation-function sweeps are outside the current experiment scope.

        "critic_hidden_size": args.critic_hidden_size,
        "critic_num_hidden_layers": args.critic_num_hidden_layers,
        "critic_activation_fn": nn.SiLU,

        # Leaving SB3 init alone means the backbone choice is the main thing changing.
        "ortho_init": True,
        # Group the spline-related settings so the registry builders do not need a long parameter list.
        "kan_kwargs": {
            "grid_size": args.grid_size,
            "spline_order": args.spline_order,
            "grid_eps": args.grid_eps,
            "grid_range": [args.grid_min, args.grid_max],
            "enable_standalone_scale_spline": True,
            "scale_noise": getattr(args, "scale_noise", 0.1),
            "use_base_branch": getattr(args, "use_base_branch", True),
            "enable_grid_updates": getattr(args, "enable_grid_updates", False),
            "grid_update_every": getattr(args, "grid_update_every", 500),
            "grid_update_until": getattr(args, "grid_update_until", 3000),
        },
}


def hold_then_fixed_window_linear_decay_schedule(initial_value: float, final_value: float, hold_steps: int, decay_duration_steps: int, total_timesteps: int,):
    """
    Create a learning-rate schedule with hold, decay, then a fixed tail.
    """
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be > 0")
    if hold_steps < 0:
        raise ValueError("hold_steps must be >= 0")
    if decay_duration_steps < 0:
        raise ValueError("decay_duration_steps must be >= 0")

    if hold_steps > total_timesteps:
        hold_steps = total_timesteps

    # Clamp the decay window so the schedule always stays inside the training horizon.
    decay_end_steps = min(hold_steps + decay_duration_steps, total_timesteps)
    effective_decay_steps = max(decay_end_steps - hold_steps, 1)

    def schedule(progress_remaining: float) -> float:
        # SB3 passes progress as 1 -> 0, but the config is easier to think about in steps.
        elapsed_steps = (1.0 - progress_remaining) * total_timesteps
        if elapsed_steps <= hold_steps:
            return initial_value
        if elapsed_steps <= decay_end_steps:
            decay_progress = (elapsed_steps - hold_steps) / effective_decay_steps
            return initial_value + decay_progress * (final_value - initial_value)
        return final_value

    return schedule


def build_ppo_kwargs(args) -> dict[str, Any]:
    """
    Build the kwargs dictionary passed directly to ``stable_baselines3.PPO``.
    """
    # PPO collects num_envs * num_steps samples before one update.
    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by "
            f"num_minibatches={args.num_minibatches}"
        )

    # SB3 uses batch_size for the minibatch size, which is easy to mix up.
    minibatch_size = batch_size // args.num_minibatches

    # Either build the piecewise schedule from config or keep a fixed learning rate.
    if getattr(args, "use_lr_schedule", True):
        learning_rate = hold_then_fixed_window_linear_decay_schedule(
            initial_value=args.learning_rate,
            final_value=args.final_learning_rate,
            hold_steps=args.lr_hold_steps,
            decay_duration_steps=args.lr_decay_steps,
            total_timesteps=args.total_timesteps,
        )
    else:
        learning_rate = args.learning_rate

    # Everything below mirrors PPO's constructor names so the runner can unpack it directly.
    return {
        "learning_rate": learning_rate, #either a float number or a function
        "n_steps": args.num_steps,
        "batch_size": minibatch_size,
        "n_epochs": args.update_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_coef,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "target_kl": args.target_kl,
        "verbose": args.verbose,
        "device": args.device,
        # PPO forwards this dict into the policy constructor.
        "policy_kwargs": build_policy_kwargs(args),
        "tensorboard_log": args.tensorboard_log if args.tensorboard_log else None,
        "seed": args.seed,
    }
