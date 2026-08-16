"""Custom SB3 policy pieces for swapping in project backbones."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from model_layer.backbones.registry import build_backbone


class CustomBackboneMlpExtractor(nn.Module):
    """
    Wrapper that lets actor and critic use different backbones.
    """

    def __init__(
        self,
        feature_dim: int,
        actor_backbone_type: str = "mlp",
        critic_backbone_type: str = "mlp",
        actor_hidden_size: int = 128,
        actor_num_hidden_layers: int = 2,
        critic_hidden_size: int = 128,
        critic_num_hidden_layers: int = 2,
        actor_activation_fn: type[nn.Module] = nn.SiLU,
        critic_activation_fn: type[nn.Module] = nn.SiLU,
        kan_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        # Keep the chosen backbone setup around because later helper methods branch on it.
        self.actor_backbone_type = actor_backbone_type
        self.critic_backbone_type = critic_backbone_type
        self.kan_kwargs = kan_kwargs or {}

        # Actor and critic counters stay separate because their forward patterns can differ.
        self.enable_grid_updates = self.kan_kwargs.get("enable_grid_updates", False)
        self.grid_update_every = self.kan_kwargs.get("grid_update_every", 500)
        self.grid_update_until = self.kan_kwargs.get("grid_update_until", 3000)
        self.grid_update_backbones = set(
            self.kan_kwargs.get(
                "grid_update_backbones",
                ["kan", "kan_no_base", "lan"],
            )
        )

        self._actor_forward_calls = 0
        self._critic_forward_calls = 0

        #generate the nets using the build_backbone function of the models/backbones/regestry.py
        self.actor_net = build_backbone(
            backbone_type=actor_backbone_type,
            obs_dim=feature_dim,
            hidden_size=actor_hidden_size,
            num_hidden_layers=actor_num_hidden_layers,
            activation_fn=actor_activation_fn,
            kan_kwargs=self.kan_kwargs,
        )

        self.critic_net = build_backbone(
            backbone_type=critic_backbone_type,
            obs_dim=feature_dim,
            hidden_size=critic_hidden_size,
            num_hidden_layers=critic_num_hidden_layers,
            activation_fn=critic_activation_fn,
            kan_kwargs=self.kan_kwargs,
        )

        # SB3 reads these attributes when it builds the action and value heads.
        self.latent_dim_pi = self.actor_net.output_dim
        self.latent_dim_vf = self.critic_net.output_dim

    def _backbone_supports_grid_updates(self, backbone_type: str) -> bool:
        """
        Checks whether this backbone type participates in adaptive grid updates.
        """
        return backbone_type in self.grid_update_backbones

    def _should_update_grid(self, forward_calls: int, backbone_type: str) -> bool:
        """
        Collect all guard conditions for triggering a grid update on this pass.
        """
        if not self.enable_grid_updates:
            return False
        if not self._backbone_supports_grid_updates(backbone_type):
            return False
        if self.grid_update_every <= 0:
            return False
        if forward_calls > self.grid_update_until:
            return False
        return forward_calls % self.grid_update_every == 0

    def _run_actor_net(self, features: torch.Tensor) -> torch.Tensor:
        """
        Run the actor backbone and request a grid update when the schedule says so.
        """
        self._actor_forward_calls += 1
        # Grid updates only make sense while training, not during evaluation or inference.
        update_grid = self.training and self._should_update_grid(
            self._actor_forward_calls,
            self.actor_backbone_type,
        )

        if update_grid:
            return self.actor_net(features, update_grid=True)
        return self.actor_net(features)

    def _run_critic_net(self, features: torch.Tensor) -> torch.Tensor:
        """
        Run the critic backbone and request a grid update when scheduled.
        """
        self._critic_forward_calls += 1
        # The critic keeps its own counter because its forward pattern can differ from the actor.
        update_grid = self.training and self._should_update_grid(
            self._critic_forward_calls,
            self.critic_backbone_type,
        )

        if update_grid:
            return self.critic_net(features, update_grid=True)
        return self.critic_net(features)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce the latent actor and critic features SB3 expects from the extractor.
        """
        latent_pi = self._run_actor_net(features)
        latent_vf = self._run_critic_net(features)
        return latent_pi, latent_vf

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Expose the actor path separately for SB3 helper calls.
        """
        return self._run_actor_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Expose the critic path separately for SB3 helper calls.
        """
        return self._run_critic_net(features)


class FlexibleActorCriticPolicy(ActorCriticPolicy):
    """
    ActorCriticPolicy that swaps SB3's shared MLP for custom actor/critic backbones.
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        *args,
        use_custom_mlp_extractor: bool = True,
        actor_backbone_type: str = "mlp",
        critic_backbone_type: str = "mlp",
        actor_hidden_size: int = 128,
        actor_num_hidden_layers: int = 2,
        critic_hidden_size: int = 128,
        critic_num_hidden_layers: int = 2,
        actor_activation_fn: type[nn.Module] = nn.SiLU,
        critic_activation_fn: type[nn.Module] = nn.SiLU,
        kan_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        #Store the backbone config before letting SB3 build the rest of the policy.
        self.use_custom_mlp_extractor = use_custom_mlp_extractor
        self.actor_backbone_type = actor_backbone_type
        self.critic_backbone_type = critic_backbone_type
        self.actor_hidden_size = actor_hidden_size
        self.actor_num_hidden_layers = actor_num_hidden_layers
        self.critic_hidden_size = critic_hidden_size
        self.critic_num_hidden_layers = critic_num_hidden_layers
        self.actor_activation_fn = actor_activation_fn
        self.critic_activation_fn = critic_activation_fn

        self.kan_kwargs = kan_kwargs or {}

        # SB3 still owns feature extraction, action heads, and optimizer setup.
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Swap in the project backbone extractor, or fall back to SB3's default MLP.
        """
        if not self.use_custom_mlp_extractor:
            super()._build_mlp_extractor()
            return

        # This is the main place where the thesis code hooks into SB3.
        self.mlp_extractor = CustomBackboneMlpExtractor(
            feature_dim=self.features_dim,
            actor_backbone_type=self.actor_backbone_type,
            critic_backbone_type=self.critic_backbone_type,
            actor_hidden_size=self.actor_hidden_size,
            actor_num_hidden_layers=self.actor_num_hidden_layers,
            critic_hidden_size=self.critic_hidden_size,
            critic_num_hidden_layers=self.critic_num_hidden_layers,
            actor_activation_fn=self.actor_activation_fn,
            critic_activation_fn=self.critic_activation_fn,
            kan_kwargs=self.kan_kwargs,
        )
