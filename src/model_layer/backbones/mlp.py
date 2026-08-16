"""Baseline MLP backbone used as the simple comparison point."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLinear(nn.Module):
    """
    Simple linear layer wrapper used by the MLP backbone.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Just the linear init.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected last dim {self.in_features}, got {x.size(-1)}"
            )

        # Flatten internally, then put the batch axes back.
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        y = F.linear(x, self.weight, self.bias)
        y = y.reshape(*original_shape[:-1], self.out_features)
        return y

    def regularization_loss(self) -> torch.Tensor:
        return self.weight.abs().mean()


class MLPBackbone(nn.Module):
    """
    Stacked MLP backbone with one activation after each linear layer.
    """

    def __init__(
        self,
        layers_hidden: list[int],
        base_activation: type[nn.Module] = nn.SiLU,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden must contain at least 2 entries")

        self._input_dim = layers_hidden[0]
        self._output_dim = layers_hidden[-1]

        self.layers = nn.ModuleList(
            [
                MLPLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                )
                for in_features, out_features in zip(
                    layers_hidden[:-1], layers_hidden[1:]
                )
            ]
        )

        # Keeping one shared activation keeps this baseline intentionally plain.
        self.activation = base_activation()

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return x

    def regularization_loss(self) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            reg = reg + layer.regularization_loss()
        return reg
