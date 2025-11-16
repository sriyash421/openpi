import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import action_chunk_broker
import pytest

import openpi.models.model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


@pytest.mark.manual
def test_infer():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "s3://openpi-assets/checkpoints/pi0_aloha_sim")

    example = aloha_policy.make_aloha_example()
    result = policy.infer(example)

    assert result["actions"].shape == (config.model.action_horizon, 14)


@pytest.mark.manual
def test_broker():
    config = _config.get_config("pi0_aloha_sim")
    policy = _policy_config.create_trained_policy(config, "s3://openpi-assets/checkpoints/pi0_aloha_sim")

    broker = action_chunk_broker.ActionChunkBroker(
        policy,
        # Only execute the first half of the chunk.
        action_horizon=config.model.action_horizon // 2,
    )

    example = aloha_policy.make_aloha_example()
    for _ in range(config.model.action_horizon):
        outputs = broker.infer(example)
        assert outputs["actions"].shape == (14,)

@pytest.mark.manual
def test_invert_actions():
    from openpi.conftest import load_checkpoint_and_dataset

    checkpoint_path = "s3://openpi-assets/checkpoints/pi0_aloha_sim"
    dataset_config_name = "pi0_aloha_sim"

    test_data = load_checkpoint_and_dataset(checkpoint_path, dataset_config_name)
    model = test_data.model
    observation = test_data.observation
    dataset_actions = test_data.dataset_actions
    train_config = test_data.train_config

    num_steps = 10
    rng = jax.random.PRNGKey(42)

    policy = _policy_config.create_trained_policy(train_config, checkpoint_path)

    obs_dict = observation.to_dict()

    def convert_image(img_array):
        img = np.asarray(img_array[0, ...])
        img = ((img + 1.0) / 2.0 * 255.0).astype(np.uint8)
        img = np.transpose(img, (2, 0, 1))
        return img

    obs_policy_format = {
        "state": np.asarray(observation.state[0, ...])[:14],
        "images": {
            "cam_high": convert_image(obs_dict["image"]["base_0_rgb"]),
            "cam_left_wrist": convert_image(obs_dict["image"]["left_wrist_0_rgb"]),
            "cam_right_wrist": convert_image(obs_dict["image"]["right_wrist_0_rgb"]),
        },
    }

    actions_dict = {
        "state": np.asarray(observation.state[0, ...]),
        "actions": np.asarray(dataset_actions[0, ...]),
    }
    actions_dict = policy._output_transform(actions_dict)
    actions_policy_format = actions_dict["actions"]

    noise_result = policy.invert_actions(obs_policy_format, actions_policy_format, num_steps=num_steps)
    noise = noise_result["noise"]

    observation_single = jax.tree.map(lambda x: x[0:1, ...], observation)
    rng, sample_rng = jax.random.split(rng)
    reconstructed_actions_model = model.sample_actions(sample_rng, observation_single, noise=noise[None, ...], num_steps=num_steps)

    reconstructed_actions_dict = {
        "state": np.asarray(observation.state[0, ...]),
        "actions": np.asarray(reconstructed_actions_model[0, ...]),
    }
    reconstructed_actions_dict = policy._output_transform(reconstructed_actions_dict)
    reconstructed_actions_policy = reconstructed_actions_dict["actions"]

    assert actions_policy_format.shape == reconstructed_actions_policy.shape
    mse = float(np.mean(np.square(actions_policy_format - reconstructed_actions_policy)))
    assert mse < 1e-2
