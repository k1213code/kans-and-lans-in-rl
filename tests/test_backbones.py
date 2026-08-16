from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from model_layer.backbones.registry import build_backbone


@pytest.mark.parametrize(
    "backbone_type",
    ["mlp", "kan", "kan_no_base", "kan_no_spline", "lan", "debug_constant"],
)
def test_backbone_registry_builds_modules_with_expected_output_shape(backbone_type: str) -> None:
    torch.manual_seed(0)
    backbone = build_backbone(
        backbone_type=backbone_type,
        obs_dim=3,
        hidden_size=4,
        num_hidden_layers=1,
        activation_fn=nn.SiLU,
        kan_kwargs={
            "grid_size": 2,
            "spline_order": 1,
            "grid_range": [-1.0, 1.0],
            "scale_noise": 0.01,
            "use_base_branch": True,
        },
    )

    x = torch.linspace(-0.5, 0.5, steps=15, dtype=torch.float32).reshape(5, 3)

    with torch.no_grad():
        y = backbone(x)

    assert y.shape == (5, 4)
    assert torch.isfinite(y).all()
    assert backbone.output_dim == 4


def test_backbone_registry_rejects_unknown_backbone() -> None:
    with pytest.raises(ValueError, match="Unknown backbone_type"):
        build_backbone(
            backbone_type="does_not_exist",
            obs_dim=3,
            hidden_size=4,
            num_hidden_layers=1,
        )
