import dataclasses
import einops
import numpy as np

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Any
from typing_extensions import override

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.transforms as transforms

# Assumed action dimension for USC WidowX (6 joints + 1 gripper)
ACTION_DIM = 7

def _decode_usc_widowx(data: dict) -> dict:
    state = np.asarray(data["state"])
    actions = np.asarray(data["actions"])

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel].
        return einops.rearrange(img, "c h w -> h w c")
    for k, v in data.items():
        if "images" in k:
            data[k] = convert_image(v)
    data["state"] = state
    data["actions"] = actions
    return data


@dataclasses.dataclass
class USCWidowXInputs(transforms.DataTransformFn):
    """Prepares USC WidowX inputs for the model.

    Assumes input keys: 'images/external', 'images/over_shoulder', 'state', 'actions'.
    The 'state' is expected to be a 7-dim vector (6 joint angles + 1 gripper state).
    The 'actions' are expected to be a 7-dim vector (6 joint actions + 1 gripper action).
    """

    action_dim: int = ACTION_DIM
    use_delta_actions: bool = False # already in delta space
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    @override
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0
        if self.action_dim != ACTION_DIM:
            raise ValueError(f"Expected action_dim={ACTION_DIM}, got {self.action_dim}")

        # Ensure expected keys are present
        required_keys = {"images/external", "images/over_shoulder", "state", "actions"}
        if not required_keys.issubset(sample.keys()):
            raise ValueError(f"Missing required keys. Found: {sample.keys()}, Required: {required_keys}")
        
        # Create inputs dict. Do not change the keys in the dict below.
        # convert images to numpy arrays
        sample = _decode_usc_widowx(sample)
        inputs = {
            "state": sample["state"],
            "image": {
                "base_0_rgb": sample["images/external"],
                "base_1_rgb": sample["images/over_shoulder"],
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "left_wrist_0_rgb": np.zeros_like(sample["images/external"]),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(sample["images/external"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "base_1_rgb": np.True_,
                "left_wrist_0_rgb": np.False_ if mask_padding else np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
            "actions": sample["actions"],
        }


        if self.use_delta_actions:
            # Convert absolute joint actions (first 6 dims) to delta actions
            delta_actions = transforms.DeltaActions(
                mask=transforms.make_bool_mask(6, -1) # Delta for joints, absolute for gripper
            )(sample)
            sample["actions"] = delta_actions["actions"]

        # Ensure proprio and actions have the correct shape
        if sample["state"].shape[-1] != self.action_dim:
             raise ValueError(f"Expected state shape (*, {self.action_dim}), got {sample['state'].shape}")
        if "actions" in sample and sample["actions"].shape[-1] != self.action_dim:
             raise ValueError(f"Expected actions shape (*, {self.action_dim}), got {sample['actions'].shape}")


        return inputs


@dataclasses.dataclass
class USCWidowXOutputs(transforms.DataTransformFn):
    """Converts model outputs back to USC WidowX action space."""

    action_dim: int = ACTION_DIM
    use_delta_actions: bool = True

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