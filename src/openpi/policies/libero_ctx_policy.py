"""LiberoInputs plus a past-attempt context image in the unused third image slot (right_wrist_0_rgb).

Keys beyond LiberoInputs:
  observation/context_image: uint8 224x224x3 tile of past-attempt frames (see onpolicy/context.py)
  observation/has_context:   bool; False (or key absent) leaves the slot zeroed and masked, i.e. plain pi05_libero input
"""
import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import libero_policy


@dataclasses.dataclass(frozen=True)
class LiberoContextInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = libero_policy.LiberoInputs(model_type=self.model_type)(data)
        if "observation/context_image" in data and bool(data.get("observation/has_context", True)):
            inputs["image"]["right_wrist_0_rgb"] = libero_policy._parse_image(data["observation/context_image"])
            inputs["image_mask"]["right_wrist_0_rgb"] = np.True_
        if "negative" in data:  # training only (onpolicy/pool.py); moved into the actions by NegativeFlagToActions
            inputs["negative"] = np.bool_(data["negative"])
        return inputs


@dataclasses.dataclass(frozen=True)
class NegativeFlagToActions(transforms.DataTransformFn):
    """Last model transform: writes the negative flag into the last action dim, which padding (7 -> 32) leaves zero,
    so it reaches the train step inside `actions` (scripts/train.py reads and clears it)."""

    def __call__(self, data: dict) -> dict:
        if "negative" in data:
            data = dict(data)
            data["actions"] = np.array(data["actions"])
            data["actions"][..., -1] = float(data.pop("negative"))
        return data
