"""Log extra backbone diagnostics when a backbone exposes them."""

from __future__ import annotations

import torch
from stable_baselines3.common.callbacks import BaseCallback


class BackboneDiagnosticsCallback(BaseCallback):
    """
    Pull optional diagnostic_metrics() from the active backbones.
    """

    def __init__(self, verbose: int = 0) -> None:
        """
        Keep the callback lightweight and only enable warnings when requested.
        """
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """
        Stay active each step so rollout-end diagnostics can still run.
        """
        return True

    def _log_metrics(self, prefix: str, module, x: torch.Tensor) -> None:
        """
        Ask one backbone for diagnostics and forward them into the SB3 logger.
        """
        if module is None or not hasattr(module, "diagnostic_metrics"):
            return

        with torch.no_grad():
            try:
                metrics = module.diagnostic_metrics(x)
            except Exception as exc:
                if self.verbose > 0:
                    print(f"[warn] {prefix} diagnostics failed: {exc}")
                return

        if not isinstance(metrics, dict):
            if self.verbose > 0:
                print(f"[warn] {prefix} diagnostics did not return a dict.")
            return

        for key, value in metrics.items():
            if value is None:
                continue
            try:
                self.logger.record(f"train/{prefix}/{key}", float(value))
            except Exception as exc:
                if self.verbose > 0:
                    print(f"[warn] Failed to log {prefix} metric '{key}': {exc}")

    def _on_rollout_end(self) -> None:
        """
        Sample the latest rollout batch and log actor/critic backbone diagnostics.
        """
        # Pull rollout observations so diagnostics are computed on the same data the update just used.
        policy = self.model.policy
        rollout_buffer = getattr(self.model, "rollout_buffer", None)
        if rollout_buffer is None:
            return

        observations = rollout_buffer.observations
        if observations is None:
            return

        obs = torch.as_tensor(observations, device=self.model.device)

        if obs.dim() >= 3:
            # Rollout buffer shape is usually [steps, envs, obs_dim], but the diagnostics code below is easier to write with one batch axis.
            obs = obs.reshape(-1, obs.shape[-1])
        elif obs.dim() != 2:
            return

        mlp_extractor = getattr(policy, "mlp_extractor", None)

        try:
            with torch.no_grad():
                # Actor and critic diagnostics should see the same feature tensor they would normally get during the forward pass.
                features = policy.extract_features(obs)
        except Exception as exc:
            if self.verbose > 0:
                print(f"[warn] Feature extraction for diagnostics failed: {exc}")
            return

        # The custom extractor keeps actor and critic backbones separate, so log them independently.
        actor_net = getattr(mlp_extractor, "actor_net", None)
        critic_net = getattr(mlp_extractor, "critic_net", None)

        self._log_metrics("actor_backbone", actor_net, features)
        self._log_metrics("critic_backbone", critic_net, features)
