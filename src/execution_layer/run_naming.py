"""Build readable run ids and folder names."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4


def generate_run_id() -> str:
    """
    Generates run id based of the datetime.
    """
    # The timestamp makes folders easier to scan by eye.
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # A short random suffix is enough to avoid collisions in practice.
    suffix = uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


def compose_run_name(env_id: str, seed: int, use_custom_mlp_extractor: bool, actor_backbone_type: str, critic_backbone_type: str, run_id: str,) -> str:
    """
    Generates the run name based of the indicated args.
    """
    actor_name = actor_backbone_type if use_custom_mlp_extractor else "sb3"
    critic_name = critic_backbone_type if use_custom_mlp_extractor else "sb3"

    return (
        f"{env_id}"
        f"__actor-{actor_name}"
        f"__critic-{critic_name}"
        f"__seed{seed}"
        f"__run-{run_id}"
    )
