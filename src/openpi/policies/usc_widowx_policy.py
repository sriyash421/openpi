import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.transforms as transforms

# Assumed action dimension for USC WidowX (6 joints + 1 gripper)
ACTION_DIM = 7


@dataclasses.dataclass
class USCWidowXInputs(transforms.DataTransform):
    """Prepares USC WidowX inputs for the model.

    Assumes input keys: 'images/external', 'images/over_shoulder', 'state', 'actions'.
    The 'state' is expected to be a 7-dim vector (6 joint angles + 1 gripper state).
    The 'actions' are expected to be a 7-dim vector (6 joint actions + 1 gripper action).
    """

    action_dim: int = ACTION_DIM
    use_delta_actions: bool = True

    @at.typed
    @override
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"Expected action_dim={ACTION_DIM}, got {self.action_dim}")

        # Ensure expected keys are present
        required_keys = {"images/external", "images/over_shoulder", "state"}
        if not required_keys.issubset(sample.keys()):
            raise ValueError(f"Missing required keys. Found: {sample.keys()}, Required: {required_keys}")

        # Rename state to proprio for consistency with model expectations
        sample["proprio"] = sample.pop("state")

        if self.use_delta_actions:
            # Convert absolute joint actions (first 6 dims) to delta actions
            delta_actions = transforms.DeltaActions(
                mask=transforms.make_bool_mask(6, -1) # Delta for joints, absolute for gripper
            )(sample)
            sample["actions"] = delta_actions["actions"]

        # Ensure proprio and actions have the correct shape
        if sample["proprio"].shape[-1] != self.action_dim:
             raise ValueError(f"Expected proprio shape (*, {self.action_dim}), got {sample['proprio'].shape}")
        if "actions" in sample and sample["actions"].shape[-1] != self.action_dim:
             raise ValueError(f"Expected actions shape (*, {self.action_dim}), got {sample['actions'].shape}")


        return sample


@dataclasses.dataclass
class USCWidowXOutputs(transforms.DataTransform):
    """Converts model outputs back to USC WidowX action space."""

    action_dim: int = ACTION_DIM
    use_delta_actions: bool = True

    @at.typed
    @override
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"Expected action_dim={ACTION_DIM}, got {self.action_dim}")

        if "actions" not in sample:
            raise ValueError("Missing 'actions' key in output sample.")

        if self.use_delta_actions:
            # Convert delta joint actions back to absolute actions
            abs_actions = transforms.AbsoluteActions(
                 mask=transforms.make_bool_mask(6, -1) # Delta for joints, absolute for gripper
            )(sample)
            sample["actions"] = abs_actions["actions"]

        # Ensure actions have the correct shape
        if sample["actions"].shape[-1] != self.action_dim:
             raise ValueError(f"Expected actions shape (*, {self.action_dim}), got {sample['actions'].shape}")

        return sample 