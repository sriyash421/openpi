import os
from typing import NamedTuple

import dataclasses
import jax.numpy as jnp
import pynvml
import pytest

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
from openpi.shared import download
from openpi.training import config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.checkpoints as _checkpoints


def set_jax_cpu_backend_if_no_gpu() -> None:
    try:
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        # No GPU found.
        os.environ["JAX_PLATFORMS"] = "cpu"


def pytest_configure(config: pytest.Config) -> None:
    set_jax_cpu_backend_if_no_gpu()


class DatasetTestData(NamedTuple):
    """Test data loaded from checkpoint and dataset."""
    model: _pi0.Pi0
    observation: _model.Observation
    dataset_actions: jnp.ndarray
    batch: dict
    train_config: _config.TrainConfig


def load_checkpoint_and_dataset(
    checkpoint_path: str,
    dataset_config_name: str,
    batch_size: int = 2,
) -> DatasetTestData:
    """Load checkpoint model and dataset batch for testing.

    Args:
        checkpoint_path: Path to checkpoint directory or S3 URL.
        dataset_config_name: Training config name to load data from.
        batch_size: Batch size for the data loader.

    Returns:
        DatasetTestData containing model, observation, actions, batch, and train_config.
    """
    config = _pi0.Pi0Config()

    checkpoint_dir = download.maybe_download(checkpoint_path)
    model = config.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    train_config = _config.get_config(dataset_config_name)
    train_config = dataclasses.replace(train_config, batch_size=batch_size)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    data_config = dataclasses.replace(
        data_config,
        norm_stats=_checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id),
    )

    dataset = _data_loader.create_dataset(data_config, train_config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=False)
    loader = _data_loader.TorchDataLoader(
        dataset, local_batch_size=batch_size, num_batches=1, num_workers=0, seed=train_config.seed
    )

    batch = next(iter(loader))
    observation = _model.Observation.from_dict(batch)
    dataset_actions = batch["actions"]

    return DatasetTestData(
        model=model,
        observation=observation,
        dataset_actions=dataset_actions,
        batch=batch,
        train_config=train_config,
    )
