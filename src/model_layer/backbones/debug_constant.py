"""Very small debug backbone that ignores the observation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConstantBackbone(nn.Module):
    """Backbone that ignores the observation and returns one fixed vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        constant_value: float = 0.0,
        trainable_constant: bool = False,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim

        # Reusing one stored vector is enough for quick wiring checks.
        const = torch.full((output_dim,), constant_value, dtype=torch.float32)
        self.register_buffer("constant", const)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected last dim {self.input_dim}, got {x.size(-1)}"
            )

        # The point is to preserve the batch shape while making the output trivial.
        out_shape = (*x.shape[:-1], self.output_dim)
        return self.constant.expand(out_shape)
