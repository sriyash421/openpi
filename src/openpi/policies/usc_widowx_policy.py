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

def _decode_usc_widowx(data: dict) -> dict:
    state = np.asarray(data["state"])
    if "actions" in data:
        actions = np.asarray(data["actions"])
    else:
        actions = None

    def convert_image(img):
        img = np.asarray(img)
        # Convert to uint8 if using float images.
        if np.issubdtype(img.dtype, np.floating):
            if np.max(img) <= 1:
                img = (255 * img).astype(np.uint8)
        # Convert from [channel, height, width] to [height, width, channel] if needed.
        if img.shape[0] == 3:
            img = einops.rearrange(img, "c h w -> h w c")
        return img
    for k, v in data.items():
        if "images" in k:
            data[k] = convert_image(v)
    data["state"] = state
    if actions is not None:
        data["actions"] = actions
    return data


@dataclasses.dataclass
class USCWidowXInputs(transforms.DataTransformFn):
    """Prepares USC WidowX inputs for the model.

    Assumes input keys: 'images/external', 'images/over_shoulder', 'state', 'actions'.
    The 'state' is expected to be a 7-dim vector (6 joint angles + 1 gripper state).
    The 'actions' are expected to be a 7-dim vector (6 joint actions + 1 gripper action).
    """

    action_dim: int
    use_delta_actions: bool = False # already in delta space
    # Determines which model will be used.
    model_type: _model.ModelType = _model.ModelType.PI0

    @override
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Ensure expected keys are present
        required_keys = {"images/external", "images/over_shoulder", "state"}
        if not required_keys.issubset(sample.keys()):
            raise ValueError(f"Missing required keys. Found: {sample.keys()}, Required: {required_keys}")
        
        # Create inputs dict. Do not change the keys in the dict below.
        # convert images to numpy arrays
        sample = _decode_usc_widowx(sample)
        inputs = {
            "state": transforms.pad_to_dim(sample["state"], self.action_dim),
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
        }
        if self.model_type == _model.ModelType.PI0:
            # no base_1_rgb for pi0
            inputs["image"]["right_wrist_0_rgb"] = inputs["image"]["base_1_rgb"]
            inputs["image_mask"]["right_wrist_0_rgb"] = inputs["image_mask"]["base_1_rgb"]
            del inputs["image"]["base_1_rgb"]
            del inputs["image_mask"]["base_1_rgb"]
        if "actions" in sample:
            inputs["actions"] = sample["actions"]
        if "prompt" in sample:
            inputs["prompt"] = sample["prompt"]



        return inputs


@dataclasses.dataclass
class USCWidowXOutputs(transforms.DataTransformFn):
    """Converts model outputs back to USC WidowX action space."""

    action_dim: int
    use_delta_actions: bool = False
    @override
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        return {"actions": np.asarray(sample["actions"][:, :7])}