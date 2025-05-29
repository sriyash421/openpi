"""
Script to convert bridge data into a single LeRobot dataset v2.0 format.

Can accept either directories containing task data (inferring trajectories within)
or explicit paths to individual trajectory directories.

Example usage (by task directories):
uv run examples/bridge/convert_usc_data_to_lerobot.py --raw-dirs /path/to/task1 /path/to/task2 --repo-id <org>/<combined-dataset-name>

Example usage (by specific trajectories):
uv run examples/usc_widowx/convert_usc_data_to_lerobot.py --traj-paths /path/to/task1/traj0 /path/to/task2/traj5 --repo-id <org>/<combined-dataset-name>
"""

import dataclasses
from pathlib import Path
import shutil
import tensorflow_datasets as tfds
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
from vila_utils.utils.decode import add_path_2d_to_img_alt_fast, add_mask_2d_to_img
from vila_utils.utils.encode import scale_path
import h5py


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    # TODO(user): Define image shape expected by LeRobot
    image_height: int = 256
    image_width: int = 256
    video_backend: str = None
    # Path and mask specific configs
    path_line_size: int = 3
    mask_pixels: int = 25
    use_paths_masks: bool = False  # Whether to process and include paths and masks


DEFAULT_DATASET_CONFIG = DatasetConfig()

RAW_DATASET_NAMES = [
    "bridge_v2",
]


def process_path_obs(sample_img, path, path_line_size=3):
    """Process path observation by drawing it onto the image."""
    height, width = sample_img.shape[:2]

    # Scale path to image size
    min_in, max_in = np.zeros(2), np.array([width, height])
    min_out, max_out = np.zeros(2), np.ones(2)
    path_scaled = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

    # Draw path
    return add_path_2d_to_img_alt_fast(sample_img, path_scaled, line_size=path_line_size)


def process_mask_obs(sample_img, mask_points, mask_pixels=25):
    """Process mask observation by applying it to the image."""
    return add_mask_2d_to_img(sample_img, mask_points, mask_pixels=mask_pixels)


def main(
    data_dir: str,
    repo_id: str,
    paths_masks_file: str = None,  # Make paths_masks_file optional
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    push_to_hub: bool = False,
) -> LeRobotDataset:
    # TODO(user): Verify motor names and count for WidowX.
    state = [
        # Example names, replace with actual motor names used in your data
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
        "gripper",
    ]
    cameras = [
        "image_0",
        "image_1",
        "image_2",
        "image_3",
    ]
    action = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    features = {
        # Corresponds to obs_dict['state'] in the TFDS script
        "observation.state": {
            "dtype": "float32",
            "shape": (len(state),),
            "names": state,
        },
        # Corresponds to policy_out['actions']
        "action": {
            "dtype": "float32",
            "shape": (len(action),),
            "names": state,
        },
        "camera_present": {
            "dtype": "bool",
            "shape": (len(cameras),),
            "names": cameras,
        },
    }

    # Add path and mask features for each camera only if use_paths_masks is True
    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": (dataset_config.image_height, dataset_config.image_width, 3),
            "names": [
                "height",
                "width",
                "channels",
            ],
        }
        if dataset_config.use_paths_masks:
            features[f"observation.path.{cam}"] = {
                "dtype": "video",
                "shape": (dataset_config.image_height, dataset_config.image_width, 3),
                "names": ["height", "width", "channels"],
            }
            features[f"observation.mask.{cam}"] = {
                "dtype": "video",
                "shape": (dataset_config.image_height, dataset_config.image_width, 3),
                "names": ["height", "width", "channels"],
            }
            features[f"observation.masked_path.{cam}"] = {
                "dtype": "video",
                "shape": (dataset_config.image_height, dataset_config.image_width, 3),
                "names": ["height", "width", "channels"],
            }

    if Path(LEROBOT_HOME / repo_id).exists():
        print(f"Removing existing dataset directory: {LEROBOT_HOME / repo_id}")
        shutil.rmtree(LEROBOT_HOME / repo_id)

    robot_type = "widowx"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=5,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

    # Load paths and masks from HDF5 file if use_paths_masks is True
    path_data = None
    path_lengths = None
    mask_data = None
    mask_lengths = None

    if dataset_config.use_paths_masks and paths_masks_file is not None:
        with h5py.File(paths_masks_file, "r") as f:
            # Loop over raw Libero datasets and write episodes to the LeRobot dataset
            for raw_dataset_name in RAW_DATASET_NAMES:
                raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
                for episode_idx, episode in enumerate(raw_dataset):
                    # Get the path and mask data for this episode from HDF5
                    episode_group = f[f"episode_{episode_idx}"]
                    path_data = episode_group["paths"][:] if "paths" in episode_group else None
                    path_lengths = episode_group["path_lengths"][:] if "paths" in episode_group else None
                    mask_data = episode_group["masks"][:] if "masks" in episode_group else None
                    mask_lengths = episode_group["mask_lengths"][:] if "masks" in episode_group else None

                    for step in episode["steps"].as_numpy_iterator():
                        frame = {
                            "observation.state": step["observation"]["state"],
                            "action": step["action"],
                            "camera_present": [True] * len(cameras),
                        }

                        # Process images, paths, and masks for each camera
                        for cam in cameras:
                            if cam in step["observation"]:
                                img = step["observation"][cam]
                                frame[f"observation.images.{cam}"] = img
                                # check for all 0 images
                                frame["camera_present"][cameras.index(cam)] = not np.all(img == 0)

                                # Process path and mask if available and enabled
                                if dataset_config.use_paths_masks and path_data is not None:
                                    # Get the current step's path
                                    step_idx = step["step_id"]
                                    if step_idx < len(path_lengths) and path_lengths[step_idx] > 0:
                                        current_path = path_data[step_idx, : path_lengths[step_idx]]
                                        # Add path to image
                                        path_img = process_path_obs(
                                            img.copy(), current_path, path_line_size=dataset_config.path_line_size
                                        )
                                        frame[f"observation.path.{cam}"] = path_img

                                        # Add mask if available
                                        if (
                                            mask_data is not None
                                            and step_idx < len(mask_lengths)
                                            and mask_lengths[step_idx] > 0
                                        ):
                                            current_mask = mask_data[step_idx, : mask_lengths[step_idx]]
                                            # Apply mask
                                            masked_img = process_mask_obs(
                                                img.copy(), current_mask, mask_pixels=dataset_config.mask_pixels
                                            )
                                            frame[f"observation.mask.{cam}"] = masked_img

                                            # Combine path and mask
                                            masked_path_img = process_path_obs(
                                                masked_img.copy(),
                                                current_path,
                                                path_line_size=dataset_config.path_line_size,
                                            )
                                            frame[f"observation.masked_path.{cam}"] = masked_path_img
                            else:
                                frame["camera_present"][cameras.index(cam)] = False
                        frame["camera_present"] = np.array(frame["camera_present"])

                        dataset.add_frame(frame)
                    dataset.save_episode(task=step["language_instruction"].decode())
    else:
        # Process without paths and masks
        for raw_dataset_name in RAW_DATASET_NAMES:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode_idx, episode in enumerate(raw_dataset):
                for step in episode["steps"].as_numpy_iterator():
                    frame = {
                        "observation.state": step["observation"]["state"],
                        "action": step["action"],
                        "camera_present": [True] * len(cameras),
                    }

                    # Process only images for each camera
                    for cam in cameras:
                        if cam in step["observation"]:
                            img = step["observation"][cam]
                            frame[f"observation.images.{cam}"] = img
                            frame["camera_present"][cameras.index(cam)] = not np.all(img == 0)
                        else:
                            frame["camera_present"][cameras.index(cam)] = False

                    dataset.add_frame(frame)
                dataset.save_episode(task=step["language_instruction"].decode())

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["widowx", "bridge-v2"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
