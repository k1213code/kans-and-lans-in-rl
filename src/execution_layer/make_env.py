"""Build Gymnasium env thunks in the format SB3 expects."""

from __future__ import annotations

import gymnasium as gym


def make_env(env_id: str, seed: int, idx: int, capture_video: bool = False):
    """
    Return a zero-argument function that builds one environment.
    """

    def thunk():
        # Rendering every parallel env would be wasteful, so only env 0 gets a video-capable mode.
        render_mode = "rgb_array" if capture_video and idx == 0 else None

        env = gym.make(env_id, render_mode=render_mode)

        # SB3 reads episode return/length from this wrapper.
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Offsetting by idx keeps runs reproducible without cloning the exact same env stream.
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)

        return env

    # SB3 wants the thunk, not an env instance created too early.
    return thunk
