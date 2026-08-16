"""Map backbone names from the config to actual modules."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from model_layer.backbones.debug_constant import ConstantBackbone
from model_layer.backbones.kan import KANBackbone
from model_layer.backbones.kan_no_base import KANNoBaseBackbone
from model_layer.backbones.kan_no_spline import KANNoSplineBackbone
from model_layer.backbones.lan import LANBackbone
from model_layer.backbones.mlp import MLPBackbone

# Keep one common factory signature so the policy code can build any backbone the same way.
BackboneBuilder = Callable[[int, int, int, type[nn.Module], dict[str, Any]], nn.Module]


def _layers_hidden(obs_dim: int, hidden_size: int, num_hidden_layers: int) -> list[int]:
    """
    Build the shared layer-width list used by all backbone types.
    """
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be >= 1")
    # All backbones in this repo use the same simple width convention.
    return [obs_dim] + [hidden_size] * num_hidden_layers


def _build_mlp(
    obs_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation_fn: type[nn.Module],
    kan_kwargs: dict[str, Any],
) -> nn.Module:
    """
    Build the plain MLP backbone used as the standard baseline.
    """
    # MLP does not use spline-specific options, so only the shared shape and activation matter here.
    return MLPBackbone(
        layers_hidden=_layers_hidden(obs_dim, hidden_size, num_hidden_layers),
        base_activation=activation_fn,
    )


def _build_kan(
    obs_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation_fn: type[nn.Module],
    kan_kwargs: dict[str, Any],
) -> nn.Module:
    """
    Build the full KAN backbone with both base and spline branches.
    """
    # Pull the KAN-specific hyperparameters from the grouped kwargs dict assembled in sb3/builders.py.
    return KANBackbone(
        layers_hidden=_layers_hidden(obs_dim, hidden_size, num_hidden_layers),
        base_activation=activation_fn,
        grid_size=kan_kwargs.get("grid_size", 5),
        spline_order=kan_kwargs.get("spline_order", 3),
        grid_eps=kan_kwargs.get("grid_eps", 0.02),
        grid_range=kan_kwargs.get("grid_range", [-1.0, 1.0]),
        enable_standalone_scale_spline=kan_kwargs.get("enable_standalone_scale_spline", True),
    )


def _build_kan_no_spline(
    obs_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation_fn: type[nn.Module],
    kan_kwargs: dict[str, Any],
) -> nn.Module:
    """
    Build the ablation variant that removes the spline component.
    """
    return KANNoSplineBackbone(
        layers_hidden=_layers_hidden(obs_dim, hidden_size, num_hidden_layers),
        base_activation=activation_fn,
        scale_base=kan_kwargs.get("scale_base", 1.0),
    )


def _build_kan_no_base(
    obs_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation_fn: type[nn.Module],
    kan_kwargs: dict[str, Any],
) -> nn.Module:
    """
    Build the ablation variant that keeps only the spline-style branch.
    """
    del activation_fn
    # This variant still needs most grid settings because the spline branch remains active.
    return KANNoBaseBackbone(
        layers_hidden=_layers_hidden(obs_dim, hidden_size, num_hidden_layers),
        grid_size=kan_kwargs.get("grid_size", 5),
        spline_order=kan_kwargs.get("spline_order", 3),
        grid_eps=kan_kwargs.get("grid_eps", 0.02),
        grid_range=kan_kwargs.get("grid_range", [-1.0, 1.0]),
        enable_standalone_scale_spline=kan_kwargs.get("enable_standalone_scale_spline", True),
        scale_noise=kan_kwargs.get("scale_noise", 0.1),
        scale_spline=kan_kwargs.get("scale_spline", 1.0),
    )


def _build_lan(
    obs_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation_fn: type[nn.Module],
    kan_kwargs: dict[str, Any],
) -> nn.Module:
    """
    Build the LAN backbone, which reuses some KAN-style grid settings.
    """
    return LANBackbone(
        layers_hidden=_layers_hidden(obs_dim, hidden_size, num_hidden_layers),
        base_activation=activation_fn,
        grid_size=kan_kwargs.get("grid_size", 5),
        spline_order=kan_kwargs.get("spline_order", 3),
        grid_range=kan_kwargs.get("grid_range", [-1.0, 1.0]),
        scale_noise=kan_kwargs.get("scale_noise", 0.1),
        scale_base=kan_kwargs.get("scale_base", 1.0),
        scale_spline=kan_kwargs.get("scale_spline", 1.0),
        enable_standalone_scale_spline=kan_kwargs.get(
            "enable_standalone_scale_spline",
            True,
        ),
        use_base_branch=kan_kwargs.get("use_base_branch", True),
        grid_eps=kan_kwargs.get("grid_eps", 0.02),
    )


def _build_debug_constant(
    obs_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation_fn: type[nn.Module],
    kan_kwargs: dict[str, Any],
) -> nn.Module:
    """
    Build a backbone that always emits the same constant features for debugging.
    """
    # Only the input/output sizes matter for this dummy module, so the remaining builder args are unused.
    del num_hidden_layers, activation_fn, kan_kwargs
    return ConstantBackbone(
        input_dim=obs_dim,
        output_dim=hidden_size,
        constant_value=0.0,
    )

# Central lookup table that maps config strings to the concrete constructor helpers above.
BACKBONE_REGISTRY: dict[str, BackboneBuilder] = {
    "mlp": _build_mlp,
    "kan": _build_kan,
    "kan_no_base": _build_kan_no_base,
    "kan_no_spline": _build_kan_no_spline,
    "lan": _build_lan,
    "debug_constant": _build_debug_constant,
}


def build_backbone(
    backbone_type: str,
    obs_dim: int,
    hidden_size: int,
    num_hidden_layers: int,
    activation_fn: type[nn.Module] = nn.SiLU,
    kan_kwargs: dict[str, Any] | None = None,
) -> nn.Module:
    """
    Build one backbone from its config name.
    """
    try:
        # Resolve the selected backbone once here so the rest of the code does not need if/else chains.
        builder = BACKBONE_REGISTRY[backbone_type]
    except KeyError as exc:
        # A clear error here is nicer than letting the policy fail much later.
        known = ", ".join(sorted(BACKBONE_REGISTRY))
        raise ValueError(f"Unknown backbone_type: {backbone_type}. Known: {known}") from exc
    # Always pass a dict onward so the individual builders can safely call .get().
    return builder(
        obs_dim,
        hidden_size,
        num_hidden_layers,
        activation_fn,
        kan_kwargs or {},
    )
