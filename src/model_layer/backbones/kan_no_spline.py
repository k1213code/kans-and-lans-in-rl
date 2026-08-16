"""Kolmogorov-Arnold Network style backbone inspiered by https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py.

Ablation to test the perfomance with only the base/ no spline component.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Single KAN layer with only the linear base branch.

    For each input-output connection, the layer learns:
    - a base linear weight applied after a fixed activation
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale_base: float = 1.0,
        base_activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.scale_base = scale_base
        self.base_activation = base_activation()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize the standard linear branch.
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected last dim {self.in_features}, got {x.size(-1)}"
            )

        # Keep the incoming batch structure so the output can match it.
        original_shape = x.shape
        # Work in 2D while computing the layer internals.
        x = x.reshape(-1, self.in_features)

        # Only the fixed base branch remains in this ablation.
        output = F.linear(self.base_activation(x), self.base_weight)

        output = output.reshape(*original_shape[:-1], self.out_features)
        return output


class KANNoSplineBackbone(nn.Module):
    """
    Stacked efficient-KAN backbone.

    Example:
        layers_hidden=[obs_dim, 128, 128]
    gives:
        obs_dim -> 128 -> 128

    Conceptually this is the KAN counterpart to ``MLPBackbone``:
    every hidden transition is a ``KANLinear`` layer, so the network keeps the
    same stacked shape as an MLP while changing what each layer computes.
    """

    def __init__(
        self,
        layers_hidden: list[int],
        scale_base: float = 1.0,
        base_activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()

        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden must contain at least 2 entries")

        self._input_dim = layers_hidden[0]
        self._output_dim = layers_hidden[-1]

        self.layers = nn.ModuleList(
            [
                KANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    scale_base=scale_base,
                    base_activation=base_activation,
                )
                for in_features, out_features in zip(
                    layers_hidden[:-1], layers_hidden[1:]
                )
            ]
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
