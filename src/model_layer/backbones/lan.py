"""MLP-style LAN backbone with learnable spline activations on the nodes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplineActivationFunction(nn.Module):
    """
    Feature-wise spline activation used after a normal MLP-style linear layer.

    This follows the spline machinery from ``KANLinear``, but adapts it
    from edge functions to node activations.
    """

    def __init__(
        self,
        num_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] = [-1.0, 1.0],
        use_base_branch: bool = True,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.use_base_branch = use_base_branch

        # KANLinear has one grid per input feature. Here the activation sees the post-linear output, so there is one grid per activated output feature.
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(num_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.spline_weight = nn.Parameter(
            torch.empty(num_features, grid_size + spline_order)
        )

        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter("spline_scaler", None)

        if use_base_branch:
            self.base_scale = nn.Parameter(torch.empty(num_features))
        else:
            self.register_parameter("base_scale", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.base_scale is not None:
            nn.init.constant_(self.base_scale, self.scale_base)

        with torch.no_grad():
            # Same idea as KANLinear: sample a small random target function on the grid and fit spline coefficients to it.
            noise = (
                (torch.rand(self.grid_size + 1, self.num_features) - 0.5)
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
                # LAN scales feature-wise spline activations, so each feature starts with
                # the same intended spline scale.
                nn.init.constant_(self.spline_scaler, self.scale_spline)



    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the B-spline basis for every activation feature."""
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.num_features}), got {tuple(x.shape)}"
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

        expected_shape = (x.size(0), self.num_features, self.grid_size + self.spline_order)
        if bases.size() != expected_shape:
            raise RuntimeError(
                f"B-spline basis shape mismatch: expected {expected_shape}, got {tuple(bases.shape)}"
            )

        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Fit spline coefficients so the spline branch matches sampled values."""
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.num_features}), got {tuple(x.shape)}"
            )
        if y.size() != (x.size(0), self.num_features):
            raise ValueError(
                f"Expected y with shape {(x.size(0), self.num_features)}, got {tuple(y.shape)}"
            )

        # Same least-squares solve as KANLinear, just without the output-edge dimension because this is a feature-wise activation.
        a = self.b_splines(x).transpose(0, 1)
        b = y.transpose(0, 1).unsqueeze(-1)
        solution = torch.linalg.lstsq(a, b).solution.squeeze(-1)

        expected_shape = (self.num_features, self.grid_size + self.spline_order,)
        if solution.size() != expected_shape:
            raise RuntimeError(
                f"curve2coeff shape mismatch: expected {expected_shape}, got {tuple(solution.shape)}"
            )

        return solution.contiguous()

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def base_branch(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_base_branch:
            return torch.zeros_like(x)
        return self.base_activation(x) * self.base_scale.unsqueeze(0)

    def spline_branch(self, x: torch.Tensor) -> torch.Tensor:
        basis = self.b_splines(x)
        return torch.sum(
            basis * self.scaled_spline_weight.unsqueeze(0),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) != self.num_features:
            raise ValueError(
                f"Expected last dim {self.num_features}, got {x.size(-1)}"
            )

        # Keep the incoming batch structure so the output can match it.
        original_shape = x.shape
        # Work in 2D while computing the layer internals.
        x = x.reshape(-1, self.num_features)

        output = self.base_branch(x) + self.spline_branch(x)
        return output.reshape(*original_shape)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Move knot locations toward the observed pre-activation distribution."""
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.num_features}), got {tuple(x.shape)}"
            )

        batch = x.size(0)

        # LAN splines are feature-wise activations, so the old spline output already has the shape expected by curve2coeff: (batch, num_features).
        old_spline_output = self.spline_branch(x)

        # Sort values feature-wise so we can pick grid points by quantiles.
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device,
            )
        ]

        # Also build a uniform grid. Mixing both grids avoids the adaptive version becoming too irregular when batches are noisy.
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=x.dtype,device=x.device,)
            .unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        # Blend data-driven knot positions with the uniform helper grid.
        grid = self.grid_eps * grid_uniform + (1.0 - self.grid_eps) * grid_adaptive
        # Add the extra boundary knots required by the spline order.
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, dtype=x.dtype, device=x.device,).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange( 1, self.spline_order + 1, dtype=x.dtype, device=x.device, ).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T.contiguous())

        if self.enable_standalone_scale_spline:
            scale = self.spline_scaler.unsqueeze(0)
            safe_scale = torch.where(
                scale.abs() > 1e-12,
                scale,
                torch.full_like(scale, 1e-12),
            )
            target = old_spline_output / safe_scale
        else:
            target = old_spline_output

        self.spline_weight.copy_(self.curve2coeff(x, target))

    def regularization_loss(
        self,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        """Spline-focused regularizer matching the KAN layer style."""
        l1_fake = self.spline_weight.abs()
        reg_activation = l1_fake.mean(dim=-1).sum()

        p = l1_fake.mean(dim=-1)
        p = p / p.sum().clamp_min(1e-12)
        reg_entropy = -torch.sum(p * p.clamp_min(1e-12).log())

        return regularize_activation * reg_activation + regularize_entropy * reg_entropy

    @torch.no_grad()
    def out_of_grid_ratio(self, x: torch.Tensor) -> float:
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.num_features}), got {tuple(x.shape)}"
            )

        core_min = self.grid[:, self.spline_order]
        core_max = self.grid[:, -(self.spline_order + 1)]
        outside = ((x < core_min.unsqueeze(0)) | (x > core_max.unsqueeze(0))).float()
        return outside.mean().item()

    @torch.no_grad()
    def base_vs_spline_ratio(self, x: torch.Tensor, eps: float = 1e-12) -> float:
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.num_features}), got {tuple(x.shape)}"
            )

        base_abs = self.base_branch(x).abs().mean()
        spline_abs = self.spline_branch(x).abs().mean()
        return (spline_abs / (base_abs + eps)).item()

    @torch.no_grad()
    def spline_base_ratio_values(self, x: torch.Tensor, eps: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"Expected x with shape (batch, {self.num_features}), got {tuple(x.shape)}"
            )

        base_abs = self.base_branch(x).abs().mean(dim=0)
        spline_abs = self.spline_branch(x).abs().mean(dim=0)
        ratio = spline_abs / (base_abs + eps)
        return ratio.reshape(-1), base_abs.reshape(-1), spline_abs.reshape(-1)


class LANLinear(nn.Module):
    """
    Simple linear layer wrapper used by the LAN backbone.

    This intentionally mirrors ``MLPLinear`` instead of importing it, so the LAN
    file shows the MLP baseline structure and the spline activation extension in
    one place.
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


class LANBackbone(nn.Module):
    """
    Stacked MLP-style backbone with a spline activation after each linear layer.
    """

    def __init__(
        self,
        layers_hidden: list[int],
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list[float] = [-1.0, 1.0],
        use_base_branch: bool = True,
        bias: bool = True,
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
                LANLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=bias,
                )
                for in_features, out_features in zip(
                    layers_hidden[:-1], layers_hidden[1:]
                )
            ]
        )

        # Unlike the plain MLP activation, the spline activation has parameters tied to the feature width, so each layer gets its own instance.
        self.activations = nn.ModuleList(
            [
                SplineActivationFunction(
                    num_features=out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    enable_standalone_scale_spline=enable_standalone_scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    use_base_branch=use_base_branch,
                )
                for out_features in layers_hidden[1:]
            ]
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor, update_grid: bool = False) -> torch.Tensor:
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            if update_grid and x.dim() != 2:
                raise ValueError(
                    "update_grid=True currently expects x to be 2D: (batch, features)"
                )

            x = layer(x)

            if update_grid:
                current_ratio = activation.out_of_grid_ratio(x)
                if current_ratio > self.grid_update_threshold:
                    print(
                        f"Layer {i}: out_of_grid_ratio={current_ratio:.4f} "
                        f"> threshold={self.grid_update_threshold:.4f} -> updating grid"
                    )
                    activation.update_grid(x)

            x = activation(x)

        return x

    def regularization_loss(
        self,
        regularize_linear: float = 0.0,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
    ) -> torch.Tensor:
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer, activation in zip(self.layers, self.activations):
            if regularize_linear > 0:
                reg = reg + regularize_linear * layer.regularization_loss()
            reg = reg + activation.regularization_loss(
                regularize_activation=regularize_activation,
                regularize_entropy=regularize_entropy,
            )
        return reg

    @torch.no_grad()
    def spline_base_ratio_values_by_layer(self, x: torch.Tensor) -> list[dict[str, torch.Tensor | int | str]]:
        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {tuple(x.shape)}")

        values = []
        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            ratio, base_abs, spline_abs = activation.spline_base_ratio_values(x)
            values.append(
                {
                    "layer": i,
                    "unit_type": "feature",
                    "ratio": ratio,
                    "base_abs": base_abs,
                    "spline_abs": spline_abs,
                }
            )
            x = activation(x)
        return values

    @torch.no_grad()
    def diagnostic_metrics(self, x: torch.Tensor) -> dict[str, float]:
        """Return per-layer and averaged diagnostics for logging callbacks."""
        if x.dim() != 2:
            raise ValueError(f"Expected x to be 2D, got {tuple(x.shape)}")

        metrics: dict[str, float] = {}
        out_of_grid_values: list[float] = []
        base_vs_spline_values: list[float] = []

        for i, (layer, activation) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)

            layer_out_of_grid = activation.out_of_grid_ratio(x)
            layer_base_vs_spline = activation.base_vs_spline_ratio(x)

            metrics[f"layer_{i}/out_of_grid_ratio"] = layer_out_of_grid
            metrics[f"layer_{i}/base_vs_spline_ratio"] = layer_base_vs_spline

            out_of_grid_values.append(layer_out_of_grid)
            base_vs_spline_values.append(layer_base_vs_spline)

            x = activation(x)

        metrics["out_of_grid_ratio"] = float(
            sum(out_of_grid_values) / len(out_of_grid_values)
        )
        metrics["base_vs_spline_ratio"] = float(
            sum(base_vs_spline_values) / len(base_vs_spline_values)
        )

        return metrics
