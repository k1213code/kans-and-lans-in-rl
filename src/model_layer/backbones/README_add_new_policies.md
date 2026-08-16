# Adding a New Backbone

This note describes the current path for adding another backbone to the project.

At the moment, this is mostly a registry-and-config task. In most cases you do not need a new policy class.


## Where a Backbone Enters the System

The relevant path is:

1. `src/interaction_layer/train_sb3_ppo.py`
2. `src/execution_layer/runner.py`
3. `src/configuration_layer/training.py`
4. `src/execution_layer/builders.py`
5. `src/model_layer/policies.py`
6. `src/model_layer/backbones/registry.py`

Backbones can currently be used in two places:

- as the custom actor backbone
- as the custom critic backbone


## Expected Backbone Interface

A backbone should:

- inherit from `nn.Module`
- expose `input_dim`
- expose `output_dim`
- implement `forward(...)`
- return tensors with the expected final latent dimension (`output_dim`)

Optional features:

- `regularization_loss()` if a future loss term should regularize this backbone
- `diagnostic_metrics(x)` if the backbone should log custom diagnostics
- `forward(x, update_grid=True)` if the backbone should support scheduled grid updates

The currently registered backbone names are:

- `mlp`
- `kan`
- `kan_no_base`
- `kan_no_spline`
- `lan`
- `debug_constant`


## Step 1: Add the Backbone File

Create a new module in:

```text
src/model_layer/backbones/
```

Use the existing `mlp.py`, `kan.py`, or `lan.py` files as references for the expected shape of the module.


## Step 2: Register the Backbone

Open:

```text
src/model_layer/backbones/registry.py
```

You need to:

- import the new backbone
- add a small builder function
- register the new name in `BACKBONE_REGISTRY`

That is the central switch used by the SB3 policy code. All builder functions use the
same signature:

```python
builder(obs_dim, hidden_size, num_hidden_layers, activation_fn, kan_kwargs)
```

The name `kan_kwargs` is historical. It is currently the shared place where
spline-related and other backbone-specific settings are passed through.


## Step 3: Expose It in the Training Config

Open:

```text
src/configuration_layer/training.py
```

If the backbone should be selectable from YAML and CLI, add it to the choices for:

- `actor_backbone_type`
- `critic_backbone_type`

If the backbone needs extra hyperparameters:

- add fields to `SB3PPOConfig`
- add matching CLI arguments
- validate that they flow correctly from config to runtime


## Step 4: Pass Backbone-Specific Settings Through the Builder Layer

Open:

```text
src/execution_layer/builders.py
```

Most backbone-specific settings currently flow through the nested `kan_kwargs`
dictionary inside `build_policy_kwargs()`. If the new backbone needs extra values,
add them there and read them in the registry builder.


## When `src/model_layer/policies.py` Needs Changes

Usually it does not.

`FlexibleActorCriticPolicy` already supports a custom actor backbone and a custom critic backbone.

You only need policy-level changes if the new backbone does not fit the current contract of:

- input -> latent feature transformation
- `nn.Module` interface
- optional diagnostics
- optional grid-update support


## Diagnostics

If the backbone exposes:

```python
diagnostic_metrics(x: torch.Tensor) -> dict[str, float]
```

then `BackboneDiagnosticsCallback` can log those values automatically.


## Grid Updates

If the backbone supports scheduled grid updates, it should accept:

```python
forward(x, update_grid=True)
```

and keep normal `forward(x)` behaviour unchanged.

It also needs to be included in the grid-update allowlist used by
`CustomBackboneMlpExtractor` in:

```text
src/model_layer/policies.py
```

If the backbone does not support that feature, no extra work is needed.


## Summary

For the current project state, adding a new backbone usually means:

- add the module
- register it
- expose it in the config
- pass any extra settings through the builder layer
