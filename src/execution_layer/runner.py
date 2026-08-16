"""Run one PPO training job from the resolved config."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from configuration_layer.training import SB3PPOConfig, parse_train_config
from execution_layer.builders import build_ppo_kwargs
from execution_layer.env_factory import build_run_name, build_vec_env, ensure_run_id, resolve_device, resolve_run_dir, set_seed
from model_layer.policies import FlexibleActorCriticPolicy
from observation_layer.backbone_diagnostics_callback import BackboneDiagnosticsCallback
from observation_layer.memory_callback import MemoryTrackingCallback
from observation_layer.time_callback import TimeTrackingCallback
from analysis_layer.training_plots import create_training_plots


def configure_torch_threads_from_env() -> None:
    """Apply optional torch thread limits passed down by experiment launchers."""
    thread_count_text = os.getenv("KAN_RL_TORCH_THREADS")
    if thread_count_text:
        torch.set_num_threads(max(1, int(thread_count_text)))

    interop_thread_count_text = os.getenv("KAN_RL_TORCH_INTEROP_THREADS")
    if interop_thread_count_text:
        torch.set_num_interop_threads(max(1, int(interop_thread_count_text)))


def write_resolved_config(args: SB3PPOConfig, run_dir: Path, run_name: str, resolved_device: str, final_model_path: Path,) -> None:
    """
    Helper function that writes the run configuration into a yaml file.
    """
    # Each run folder should contain the exact settings that produced it, plus runtime values that only exist after startup.
    resolved = asdict(args)
    resolved["run_name"] = run_name
    resolved["run_dir"] = str(run_dir)
    resolved["resolved_device"] = resolved_device
    resolved["final_model_path"] = str(final_model_path)

    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved, handle, sort_keys=True)


def run_training(args: SB3PPOConfig) -> Path:
    """
    Function that sets up the enviroment, the model and starts the training. Afterward its saves the model and a progress csv with the training data and generates plots for some of the data.
    """
    configure_torch_threads_from_env()

    # Enviroment creation and general set up
    set_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    resolved_device = resolve_device(args)
    ensure_run_id(args)
    run_name = build_run_name(args)
    run_dir = resolve_run_dir(args, run_name)

    run_dir.mkdir(parents=True, exist_ok=True)
    env = build_vec_env(args)

    #obtain the kwargs for the PPO setup
    ppo_kwargs = build_ppo_kwargs(args)
    ppo_kwargs["device"] = resolved_device

    # print configuration of the run before it starts
    print(f"Run: {run_name}")
    print(f"Device: {resolved_device}")
    print(
        f"Torch threads: intra={torch.get_num_threads()}, "
        f"interop={torch.get_num_interop_threads()}"
    )
    print(f"Env: {args.env_id}")
    print(f"SB3 policy: {FlexibleActorCriticPolicy.__name__}")
    print(
        f"actor_net="
        f"{args.actor_backbone_type if args.use_custom_mlp_extractor else 'standard_sb3'}"
    )
    print(
        f"critic_net="
        f"{args.critic_backbone_type if args.use_custom_mlp_extractor else 'standard_sb3'}"
    )
    print(
        f"num_envs={args.num_envs}, n_steps={args.num_steps}, "
        f"total_timesteps={args.total_timesteps}"
    )

    if args.save_path:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Default to the run directory so everything stays in one place.
        save_path = run_dir / "model.zip"
        save_path.parent.mkdir(parents=True, exist_ok=True)

    write_resolved_config(
        args=args,
        run_dir=run_dir,
        run_name=run_name,
        resolved_device=resolved_device,
        final_model_path=save_path,
    )

    # This is the point where SB3 actually instantiates the policy and the chosen backbones.
    model = PPO(
        FlexibleActorCriticPolicy, # Custom backbone or standard PPO
        env, # Enviroment for training
        **ppo_kwargs, # PPO arguments ** allows for flexible number of keyword arguments
    )

    # Set up logging. Keeping all logger outputs inside the run folder makes post-processing easier later.
    logger = configure(str(run_dir), ["stdout", "csv"]) #stdout for live terminal updates, csv to extract data for curves etc after training
    model.set_logger(logger)

    callbacks = []
    if args.track_memory:
        callbacks.append(MemoryTrackingCallback(log_every_n_calls=args.memory_log_every, verbose=0))
    if args.track_time:
        callbacks.append(TimeTrackingCallback(log_every_n_calls=args.time_log_every, verbose=0))
    callbacks.append(BackboneDiagnosticsCallback(verbose=1))

    #not planned to be used but potentially usefull for trouble shooting
    if args.checkpoint_freq > 0:
        # If the user picked a custom save path, keep checkpoints next to it too.
        save_dir = Path(args.save_path) if args.save_path else run_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                # The filename itself should identify the setup that produced it.
                save_path=str(save_dir),
                name_prefix=(
                    f"actor_{args.actor_backbone_type if args.use_custom_mlp_extractor else 'sb3'}"
                    f"__critic_{args.critic_backbone_type if args.use_custom_mlp_extractor else 'sb3'}_ppo"
                ),
            )
        )

    callback = CallbackList(callbacks) if callbacks else None

    #starts the training
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True,
        tb_log_name=run_name,
    )

    #generate the plots for some of the logged data using visualisation/trainig_ploty.py
    if args.make_plots:
        # The training logs are already written at this point, so plotting can stay separate from the learning loop.
        create_training_plots(
            progress_csv=run_dir / "progress.csv",
            outdir=run_dir / "plots",
            smooth_window=args.smooth_window,
        )

    model.save(str(save_path))
    print(f"Saved final model to: {save_path}")
    # Closing explicitly avoids hanging env resources after training.
    env.close()
    return save_path


def main(argv: list[str] | None = None) -> Path:
    args = parse_train_config(argv) #obtains teh args from the handed list of arguments using config/training.py
    return run_training(args)
