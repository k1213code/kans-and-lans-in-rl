"""Small helpers used before PPO training starts."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from configuration_layer.training import SB3PPOConfig
from execution_layer.make_env import make_env
from execution_layer.run_naming import compose_run_name, generate_run_id
from utility_layer.paths import resolve_output_path


def set_seed(seed: int) -> None:
    """
    Sets the seeds for pythons, numpys and torches RNG
    """
    # Keep the usual random sources aligned across reruns.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_vec_env(args: SB3PPOConfig):
    """
    Builds vec env used in the later PPO model setup.
    """
    env_fns = []

    for i in range(args.num_envs):
        def _thunk(i=i):
            # The default arg keeps each closure tied to its own env index.
            env = make_env(args.env_id, args.seed, i, capture_video=False)()
            # Monitor is where SB3 gets episode return/length logging from.
            env = Monitor(env)
            return env

        env_fns.append(_thunk)

    # DummyVecEnv is preverable here since it lends mor towards reproducibility and determinism
    env = DummyVecEnv(env_fns)
    # VecMonitor keeps the vectorized wrapper from losing the monitor stats.
    env = VecMonitor(env)
    return env


def resolve_device(args: SB3PPOConfig) -> str:
    """
    Sets the device that is to be used, automatically chooses cuda if available and auto is choosen.
    (cuda is actually slower for these small models)
    """
    # CLI/device config should win over any automatic guess.
    if args.device != "auto":
        return args.device
    # Otherwise only use CUDA when the machine has it and the config allows it.
    return "cuda" if torch.cuda.is_available() and args.cuda else "cpu"


def ensure_run_id(args: SB3PPOConfig) -> str:
    """
    Generates a run id if non is given.
    """
    if not args.run_id:
        args.run_id = generate_run_id()
    return args.run_id


def build_run_name(args: SB3PPOConfig) -> str:
    """
    Builds the run name using training/run_naming.py
    """
    # This keeps folder names readable without relying only on timestamps.
    run_id = ensure_run_id(args)
    return compose_run_name(
        env_id=args.env_id,
        seed=args.seed,
        use_custom_mlp_extractor=args.use_custom_mlp_extractor,
        actor_backbone_type=args.actor_backbone_type,
        critic_backbone_type=args.critic_backbone_type,
        run_id=run_id,
    )


def resolve_run_dir(args: SB3PPOConfig, run_name: str) -> Path:
    return resolve_output_path(args.output_root) / "runs" / run_name
