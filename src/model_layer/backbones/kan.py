"""Kolmogorov-Arnold Network style backbone inspiered by https://github.com/Blealtan/efficient-kan/blob/master/src/efficient_kan/kan.py.

KAN differs from the MLP baseline by learning spline functions on edges rather
than relying only on a fixed node activation. Each layer combines a standard
base path with a spline correction path evaluated on a learnable knot grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Single KAN layer with a linear base branch and spline edge functions.

    For each input-output connection, the layer learns:
    - a base linear weight applied after a fixed activation
    - spline coefficients over a knot grid that add a learned nonlinear
      correction

    The final output is ``base_output + spline_output``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] = [-1.0, 1.0],
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        # Each input feature gets its own knot grid. Extra boundary knots are included because B-splines of order > 0 need them to evaluate bases.
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.register_parameter("spline_scaler", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize the standard linear branch.
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            # Sample a small random target function on the grid and fit spline coefficients to it. This gives the spline branch a gentle start instead of a large arbitrary signal.
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)- 0.5)
                * self.scale_noise
                / self.grid_size
            )

            coeff = self.curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order],
                noise,
            )

            self.spline_weight.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * coeff
            )

            if self.enable_standalone_scale_spline:
                # KAN learns spline functions per input-output edge, so the scaler is a weight matrix and can use weight-style random initialization
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline basis for every input feature independently."""
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )

        grid = self.grid
        # Add a trailing dimension so each value can be compared against all knot intervals of its feature-specific grid.
        x = x.unsqueeze(-1)

        # Start with piecewise-constant interval indicators.
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.spline_order + 1):
            # Recursively upgrade the basis from order k-1 to order k.
            left_num = x - grid[:, : -(k + 1)]
            left_den = grid[:, k:-1] - grid[:, : -(k + 1)]
            right_num = grid[:, k + 1 :] - x
            right_den = grid[:, k + 1 :] - grid[:, 1:(-k)]

            left = torch.where(
                left_den != 0,
                left_num / left_den,
                torch.zeros_like(left_num),
            )
            right = torch.where(
                right_den != 0,
                right_num / right_den,
                torch.zeros_like(right_num),
            )

            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]

        expected_shape = (x.size(0), self.in_features, self.grid_size + self.spline_order)
        if bases.size() != expected_shape:
            raise RuntimeError(
                f"B-spline basis shape mismatch: expected {expected_shape}, got {tuple(bases.shape)}"
            )

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fit spline coefficients so the spline branch matches sampled values."""
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )
        if y.size() != (x.size(0), self.in_features, self.out_features):
            raise ValueError(
                f"Expected y with shape {(x.size(0), self.in_features, self.out_features)}, got {tuple(y.shape)}"
            )

        # Build the basis matrix A and solve A * coeff ~= y in least squares form for each input feature independently.
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1)
        solution = torch.linalg.lstsq(a, b).solution
        result = solution.permute(2, 0, 1)

        expected_shape = (self.out_features, self.in_features, self.grid_size + self.spline_order,)
        if result.size() != expected_shape:
            raise RuntimeError(
                f"curve2coeff shape mismatch: expected {expected_shape}, got {tuple(result.shape)}"
            )

        return result.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        # Some variants learn a separate multiplier for the spline branch.
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected last dim {self.in_features}, got {x.size(-1)}"
            )

        # Keep the incoming batch structure so the output can match it.
        original_shape = x.shape
        # Work in 2D while computing the layer internals.
        x = x.reshape(-1, self.in_features)

        # The fixed base branch keeps a familiar linear path, while the spline branch adds input-dependent nonlinear corrections on each edge.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # The spline basis expands each input feature into local basis values. Those basis values are then linearly combined by learned coefficients.
        spline_output = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )

        # KAN combines both branches to get the final layer output.
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Move knot locations toward the observed input distribution.

        The spline coefficients are re-fitted after the move so the represented
        spline function changes as little as possible.
        """
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )

        batch = x.size(0)

        # Evaluate the current spline branch before moving the grid, so we can approximately preserve the same function afterward.
        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)

        # Sort values feature-wise so we can pick grid points by quantiles.
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        # Also build a uniform grid. Mixing both grids avoids the adaptive version becoming too irregular when batches are noisy.
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device)
            .unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # Blend data-driven knot positions with the uniform helper grid.
        grid = self.grid_eps * grid_uniform + (1.0 - self.grid_eps) * grid_adaptive
        # Add the extra boundary knots required by the spline order.
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        old_grid = self.grid.detach().clone()
        new_grid = grid.T.contiguous()

        print("Old grid:")
        print(old_grid)
        print("New grid:")
        print(new_grid)
        print("Grid delta:")
        print(new_grid - old_grid)

        # Install the new grid and then re-fit coefficients on that grid.
        self.grid.copy_(new_grid)
        self.spline_weight.copy_(self.curve2coeff(x, unreduced_spline_output))
        print("grid updated")

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        """Spline-focused regularizer used to discourage overly sharp solutions."""
        # Mean absolute coefficient size: encourages smaller spline weights.
        l1_fake = self.spline_weight.abs().mean(-1)
        reg_activation = l1_fake.sum()
        # Entropy term: discourages all spline activity from collapsing into
        # only a few connections.
        p = l1_fake / reg_activation.clamp_min(1e-12)
        reg_entropy = -torch.sum(p * p.clamp_min(1e-12).log())
        return regularize_activation * reg_activation + regularize_entropy * reg_entropy

    @torch.no_grad()
    def out_of_grid_ratio(self, x: torch.Tensor) -> float:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )

        # Ignore the extra boundary knots and only look at the "real" grid
        # range that the spline was meant to cover.
        core_min = self.grid[:, self.spline_order]
        core_max = self.grid[:, -(self.spline_order + 1)]

        # Ratio of values that fall outside the usable spline support.
        outside = ((x < core_min.unsqueeze(0)) | (x > core_max.unsqueeze(0))).float()
        return outside.mean().item()

    @torch.no_grad()
    def base_vs_spline_ratio(self, x: torch.Tensor, eps: float = 1e-12) -> float:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )

        # Compare how large the spline branch is relative to the base branch.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.scaled_spline_weight.reshape(self.out_features, -1),
        )

        base_abs = base_output.abs().mean()
        spline_abs = spline_output.abs().mean()

        return (spline_abs / (base_abs + eps)).item()

    @torch.no_grad()
    def spline_base_ratio_values(self, x: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.in_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )

        base_edges = self.base_activation(x).unsqueeze(1) * self.base_weight.unsqueeze(0)
        spline_edges = torch.einsum("bik,oik->boi", self.b_splines(x), self.scaled_spline_weight)
        base_abs = base_edges.abs().mean(dim=0)
        spline_abs = spline_edges.abs().mean(dim=0)
        ratio = spline_abs / (base_abs + eps)
        return ratio.reshape(-1), base_abs.reshape(-1), spline_abs.reshape(-1)


class KANBackbone(nn.Module):
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
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] = [-1.0, 1.0],
        enable_standalone_scale_spline: bool = True,
        grid_update_threshold: float = 0.1,
    ) -> None:
        super().__init__()

        if len(layers_hidden) < 2:
            raise ValueError("layers_hidden must contain at least 2 entries")

        self._input_dim = layers_hidden[0]
        self._output_dim = layers_hidden[-1]
        self.grid_update_threshold = grid_update_threshold

        self.layers = nn.ModuleList(
            [
                KANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    enable_standalone_scale_spline=enable_standalone_scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
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
        for i, layer in enumerate(self.layers):
            if update_grid:
                if x.dim() != 2:
                    raise ValueError(
                        "update_grid=True currently expects x to be 2D: (batch, features)"
                    )

                # Grid updates are optional and only happen when incoming
                # activations drift too far outside the current spline support.
                current_ratio = layer.out_of_grid_ratio(x)
                if current_ratio > self.grid_update_threshold:
                    print(
                        f"Layer {i}: out_of_grid_ratio={current_ratio:.4f} "
                        f"> threshold={self.grid_update_threshold:.4f} -> updating grid"
                    )
                    layer.update_grid(x)

            # Feed the current activations into the next KAN layer.
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0,) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            # Add the regularization contribution from every KAN layer.
            reg = reg + layer.regularization_loss(
                regularize_activation=regularize_activation,
                regularize_entropy=regularize_entropy,
            )
        return reg

    @torch.no_grad()
    def spline_base_ratio_values_by_layer(self, x: torch.Tensor) -> list[dict[str, torch.Tensor | int | str]]:
        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {tuple(x.shape)}")

        values = []
        for i, layer in enumerate(self.layers):
            ratio, base_abs, spline_abs = layer.spline_base_ratio_values(x)
            values.append(
                {
                    "layer": i,
                    "unit_type": "edge",
                    "ratio": ratio,
                    "base_abs": base_abs,
                    "spline_abs": spline_abs,
                }
            )
            x = layer(x)
        return values

    @torch.no_grad()
    def diagnostic_metrics(self, x: torch.Tensor) -> dict[str, float]:
        """Return per-layer and averaged diagnostics for logging callbacks."""
        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {tuple(x.shape)}")

        metrics: dict[str, float] = {}
        out_of_grid_values: list[float] = []
        base_vs_spline_values: list[float] = []

        for i, layer in enumerate(self.layers):
            # Log diagnostics before applying the layer, so the metrics describe
            # the current input distribution seen by that layer.
            layer_out_of_grid = layer.out_of_grid_ratio(x)
            layer_base_vs_spline = layer.base_vs_spline_ratio(x)

            metrics[f"layer_{i}/out_of_grid_ratio"] = layer_out_of_grid
            metrics[f"layer_{i}/base_vs_spline_ratio"] = layer_base_vs_spline

            out_of_grid_values.append(layer_out_of_grid)
            base_vs_spline_values.append(layer_base_vs_spline)

            x = layer(x)

        # Also expose simple averages for dashboard-friendly monitoring.
        metrics["out_of_grid_ratio"] = float(sum(out_of_grid_values) / len(out_of_grid_values))
        metrics["base_vs_spline_ratio"] = float(sum(base_vs_spline_values) / len(base_vs_spline_values))

        return metrics
