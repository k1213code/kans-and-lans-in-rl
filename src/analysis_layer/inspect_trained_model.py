"""Inspect a trained SB3 PPO policy with one example observation.

Torchinfo: https://pypi.org/project/torchinfo/
Torchview: https://pypi.org/project/torchview/
"""

# run: python src\interaction_layer\inspect_trained_model.py src\output_layer\outputs\runs\HalfCheetah-v5__actor-kan__critic-kan__seed2025__run-paper\model.zip --graph-path src\output_layer\outputs\model_graph.svg

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from torchinfo import summary
from torchview import draw_graph

# Loading a saved model will fail if the custom policy class is not importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_layer.utils import register_legacy_module_aliases


register_legacy_module_aliases()


def parse_args() -> argparse.Namespace:
    """
    Read the small set of CLI options needed for model inspection.
    """
    parser = argparse.ArgumentParser(description="Inspect a saved SB3 PPO model.")
    parser.add_argument("model_path", type=str, help="Path to the saved model zip.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading, e.g. cpu or cuda.")
    parser.add_argument(
        "--graph-path",
        type=str,
        default="",
        help="Optional output file for the torchview graph, e.g. src/output_layer/outputs/model_graph.svg.",
    )
    return parser.parse_args()


def make_demo_observation(observation_space: spaces.Box) -> np.ndarray:
    """
    Build one fixed example observation from the Box bounds.
    """
    # A midpoint-style observation is enough for inspection and avoids random noise in the report.
    observation = np.zeros(observation_space.shape, dtype=np.float32)
    finite_bounds = np.isfinite(observation_space.low) & np.isfinite(observation_space.high)
    observation[finite_bounds] = (
        (observation_space.low[finite_bounds] + observation_space.high[finite_bounds]) / 2.0
    ).astype(np.float32)
    return observation.astype(observation_space.dtype)


def format_tensor_stats(name: str, tensor: torch.Tensor) -> str:
    """
    Return one compact text line with the key stats of a forward output tensor.
    """
    detached = tensor.detach()
    shape = tuple(detached.shape)
    dtype = str(detached.dtype).replace("torch.", "")
    device = str(detached.device)

    if detached.numel() == 0:
        return f"{name}: shape={shape}, dtype={dtype}, device={device}, empty=True"

    if detached.is_floating_point() or detached.is_complex():
        detached = detached.float()
        return (
            f"{name}: shape={shape}, dtype={dtype}, device={device}, "
            f"mean={detached.mean().item():.6g}, std={detached.std(unbiased=False).item():.6g}, "
            f"min={detached.min().item():.6g}, max={detached.max().item():.6g}"
        )

    return (
        f"{name}: shape={shape}, dtype={dtype}, device={device}, "
        f"min={detached.min().item()}, max={detached.max().item()}"
    )


def save_graph(policy: torch.nn.Module, obs_tensor: Any, graph_path: str) -> str:
    """
    Render the policy graph from the same example input used for the forward pass.
    """
    if not graph_path:
        return "No graph file requested."

    output_path = Path(graph_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_graph = draw_graph(
        policy,
        input_data=(obs_tensor,),
        expand_nested=True,
        device=str(policy.device),
        graph_name=output_path.stem,
    )

    graph_format = output_path.suffix.lower().lstrip(".") or "svg"
    model_graph.visual_graph.render(
        filename=output_path.stem,
        directory=str(output_path.parent),
        format=graph_format,
        cleanup=True,
    )
    return f"Saved torchview graph to: {output_path}"


def build_report(args: argparse.Namespace) -> str:
    """
    Load the model, run one example forward pass, and format the inspection output.
    """
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PPO.load(str(model_path), device=args.device)
    policy = model.policy
    policy.eval()

    observation_space = model.observation_space
    if not isinstance(observation_space, spaces.Box):
        raise TypeError(
            "This inspection script is simplified for Box observation spaces, "
            f"but got: {type(observation_space).__name__}"
        )

    # Let the policy convert the NumPy observation so shapes/devices match what SB3 expects.
    raw_obs = make_demo_observation(observation_space)
    obs_tensor, vectorized = policy.obs_to_tensor(raw_obs)

    with torch.no_grad():
        actions, values, log_prob = policy(obs_tensor, deterministic=True)

    report: list[str] = [
        "Model Inspection Report",
        "",
        f"model_path: {model_path.resolve()}",
        f"algorithm_class: {model.__class__.__name__}",
        f"policy_class: {policy.__class__.__name__}",
        f"device: {model.device}",
        f"observation_shape: {observation_space.shape}",
        f"observation_dtype: {observation_space.dtype}",
        f"action_space: {model.action_space}",
        f"vectorized_input_detected_by_policy: {vectorized}",
        "",
        "Forward Output",
        format_tensor_stats("actions", actions),
        format_tensor_stats("values", values),
        format_tensor_stats("log_prob", log_prob),
        "",
        "torchinfo Summary",
        str(
            summary(
                policy,
                input_data=(obs_tensor,),
                depth=6,
                col_names=("input_size", "output_size", "num_params", "trainable"),
                row_settings=("var_names", "depth"),
                verbose=0,
                device=str(policy.device),
            )
        ),
        "",
        "torchview",
        "",
    ]

    try:
        report.append(save_graph(policy, obs_tensor, args.graph_path))
    except Exception as exc:
        # In practice this is usually a missing Graphviz install.
        report.append(
            "torchview graph creation failed. "
            f"This often means Graphviz is missing or the model uses an unsupported path. Error: {exc}"
        )

    return "\n".join(report)


def main() -> None:
    """
    Parse arguments, inspect the model, and print the result.
    """
    print(build_report(parse_args()))


if __name__ == "__main__":
    main()
