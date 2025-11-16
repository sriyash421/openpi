from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._action_to_noise = nnx_utils.module_jit(model.action_to_noise)
        self._rng = rng if rng is not None else jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def invert_actions(self, obs: dict, actions: np.ndarray, num_steps: int = 100) -> dict:
        # Convert dict to Observation format first to get action_dim from state shape
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        # Get action_dim from observation state (state is padded to action_dim)
        action_dim = observation.state.shape[-1]

        # Reverse output transforms to convert actions from output format to model format
        # Actions from infer() are in output format (e.g., 14D for Aloha), but model expects model format (32D)
        actions_model_format = self._reverse_output_transforms(actions, action_dim)

        actions_batched = jnp.asarray(actions_model_format)[np.newaxis, ...]
        noise = self._action_to_noise(observation, actions_batched, num_steps)
        # Unbatch and convert to np.ndarray.
        noise = np.asarray(noise[0, ...])
        return {"noise": noise}

    def _reverse_output_transforms(self, actions: np.ndarray, action_dim: int) -> np.ndarray:
        """Reverse output transforms to convert actions from output format back to model format."""
        # Only reverse AlohaOutputs transform (slice + encode), skip others like Unnormalize
        # CompositeTransform stores transforms in .transforms attribute
        if isinstance(self._output_transform, _transforms.CompositeTransform):
            transforms_to_reverse = self._output_transform.transforms
        else:
            # If it's a single transform, wrap it in a list
            transforms_to_reverse = [self._output_transform]

        # Reverse only AlohaOutputs transform
        from openpi.policies import aloha_policy
        for transform in reversed(transforms_to_reverse):
            if isinstance(transform, aloha_policy.AlohaOutputs):
                # Reverse _encode_actions by applying _encode_actions_inv
                actions = aloha_policy._encode_actions_inv(actions, adapt_to_pi=transform.adapt_to_pi)
                # Pad back to model action_dim (AlohaOutputs slices to 14, need to pad to 32)
                actions = _transforms.pad_to_dim(actions, action_dim)
                break

        return actions


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
