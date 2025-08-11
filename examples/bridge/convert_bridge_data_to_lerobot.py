"""
Script to convert bridge data into a single LeRobot dataset v2.0 format.

Can accept either directories containing task data (inferring trajectories within)
or explicit paths to individual trajectory directories.

Example usage (by task directories):
uv run examples/bridge/convert_usc_data_to_lerobot.py --raw-dirs /path/to/task1 /path/to/task2 --repo-id <org>/<combined-dataset-name>

Example usage for path masks:
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /data/shared/openx_rlds_data/ --repo_id jesbu1/bridge_v2_lerobot_pathmask --paths-masks-file ~/VILA/test_bridge_labeling_5x/bridge_paths_masks_merged.h5 --push_to_hub
"""

import dataclasses
from pathlib import Path
import shutil
import tensorflow_datasets as tfds
import numpy as np
try:
    # for older lerobot versions before 2.0.0
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    OLD_LEROBOT = True
except ImportError:
    # newer lerobot versions use HF_LEROBOT_HOME instead of LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
    OLD_LEROBOT = False
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro
from vila_utils.utils.decode import add_path_2d_to_img_alt_fast, add_mask_2d_to_img
from vila_utils.utils.encode import scale_path, smooth_path_rdp
import h5py


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    image_writer_processes: int = 0
    image_writer_threads: int = 16
    # TODO(user): Define image shape expected by LeRobot
    image_height: int = 224
    image_width: int = 224
    video_backend: str = None
    # Path and mask specific configs
    path_line_size: int = 2
    use_paths_masks: bool = True  # Whether to process and include paths and masks
    apply_rdp: bool = False # Whether to apply RDP to the path and mask output from the VLM


DEFAULT_DATASET_CONFIG = DatasetConfig()

RAW_DATASET_NAMES = [
    "bridge_v2",
]


def process_path_obs(sample_img, path, path_line_size=3, apply_rdp=False):
    """Process path observation by drawing it onto the image."""
    height, width = sample_img.shape[:2]

    # Scale path to image size
    min_in, max_in = np.zeros(2), np.array([width, height])
    min_out, max_out = np.zeros(2), np.ones(2)
    path_scaled = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

    if apply_rdp:
        length_before = len(path_scaled)
        path_scaled = smooth_path_rdp(path_scaled, tolerance=0.05)
        length_after = len(path_scaled)
        if length_before > length_after:
            print(f"RDP reduced path length from {length_before} to {length_after}")
    # Draw path
    return add_path_2d_to_img_alt_fast(sample_img, path_scaled, line_size=path_line_size)


def process_mask_obs(sample_img, mask_points, mask_pixels=25, scale_mask=False, apply_rdp=False):
    """Process mask observation by applying it to the image."""
    if scale_mask:
        height, width = sample_img.shape[:2]

        # Scale mask points to image size
        min_in, max_in = np.zeros(2), np.array([width, height])
        min_out, max_out = np.zeros(2), np.ones(2)
        mask_points_scaled = scale_path(mask_points, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
    else:
        mask_points_scaled = mask_points

    if apply_rdp:
        length_before = len(mask_points_scaled)
        mask_points_scaled = smooth_path_rdp(mask_points_scaled, tolerance=0.05)
        length_after = len(mask_points_scaled)
        if length_before > length_after:
            print(f"RDP reduced mask length from {length_before} to {length_after}")

    return add_mask_2d_to_img(sample_img, mask_points_scaled, mask_pixels=mask_pixels)


def main(
    data_dir: str,
    repo_id: str,
    paths_masks_file: str = None,  # Make paths_masks_file optional
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    push_to_hub: bool = False,
    mask_ratio: float = 0.08,
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
        #"image_1",
        #"image_2",
        #"image_3",
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
    if dataset_config.use_paths_masks and paths_masks_file is not None:
        with h5py.File(paths_masks_file, "r", swmr=True) as path_masks_h5:
            # Loop over raw Libero datasets and write episodes to the LeRobot dataset
            for raw_dataset_name in RAW_DATASET_NAMES:
                raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
                for episode_idx, episode in enumerate(raw_dataset):
                    # Get the path and mask data for this episode from HDF5
                    if f"episode_{episode_idx}" in path_masks_h5:
                        episode_group = path_masks_h5[f"episode_{episode_idx}"]
                    else:
                        continue
                    next_path_timestep_idx = 0
                    next_mask_timestep_idx = 0


                    #mask_ratio = np.random.uniform(dataset_config.mask_ratio_min, dataset_config.mask_ratio_max)

                    for step_idx, step in enumerate(episode["steps"].as_numpy_iterator()):
                        frame = {
                            "observation.state": step["observation"]["state"],
                            "action": step["action"],
                            "camera_present": [True] * len(cameras),
                        }

                        for cam in cameras:
                            # Get the path and mask data for this camera from HDF5
                            path_data = episode_group[f"{cam}_paths"][:] if f"{cam}_paths" in episode_group else None
                            path_lengths = (
                                episode_group[f"{cam}_path_lengths"][:]
                                if f"{cam}_path_lengths" in episode_group
                                else None
                            )
                            mask_data = episode_group[f"{cam}_masks"][:] if f"{cam}_masks" in episode_group else None
                            mask_lengths = (
                                episode_group[f"{cam}_mask_lengths"][:]
                                if f"{cam}_mask_lengths" in episode_group
                                else None
                            )
                            # Timesteps correspond to which steps have the path and masks
                            path_timesteps = (
                                episode_group[f"{cam}_path_timesteps"][:]
                                if f"{cam}_path_timesteps" in episode_group
                                else None
                            )
                            mask_timesteps = (
                                episode_group[f"{cam}_mask_timesteps"][:]
                                if f"{cam}_mask_timesteps" in episode_group
                                else None
                            )

                            if cam in step["observation"]:
                                img = step["observation"][cam]
                                frame[f"observation.images.{cam}"] = img
                                # check for all 0 images
                                frame["camera_present"][cameras.index(cam)] = not np.all(img == 0)
                                # Skip if any of the path or mask data is not available or if the camera is not present
                                if (
                                    any(data is None for data in [path_data, path_lengths, mask_data, mask_lengths])
                                    or frame["camera_present"][cameras.index(cam)] is False
                                ):
                                    frame[f"observation.path.{cam}"] = np.zeros_like(img)
                                    frame[f"observation.masked_path.{cam}"] = np.zeros_like(img)
                                    continue

                                # Process path and mask if available and enabled
                                if dataset_config.use_paths_masks and path_data is not None:
                                    # Get the current step's path
                                    # because we query the path and mask data for each step but it's only generated every N steps in the HDF5, we check if the step_idx has reached the next timestep
                                    if step_idx == path_timesteps[next_path_timestep_idx % len(path_timesteps)] and next_path_timestep_idx < len(path_timesteps):
                                        next_path_timestep_idx += 1
                                    if step_idx == mask_timesteps[next_mask_timestep_idx % len(mask_timesteps)] and next_mask_timestep_idx < len(mask_timesteps):
                                        next_mask_timestep_idx += 1
                                    

                                    current_path_idx = next_path_timestep_idx - 1
                                    current_mask_idx = next_mask_timestep_idx - 1

                                    current_path = path_data[current_path_idx, : path_lengths[current_path_idx]]
                                    # Add path to image
                                    path_img = process_path_obs(
                                        img.copy(), current_path, path_line_size=dataset_config.path_line_size, apply_rdp=dataset_config.apply_rdp
                                    )
                                    frame[f"observation.path.{cam}"] = path_img

                                    # Add mask if available
                                    if (
                                        mask_data is not None
                                    ):
                                        current_mask = mask_data[current_mask_idx, : mask_lengths[current_mask_idx]]
                                        # Apply mask
                                        height, width = img.shape[:2]
                                        masked_img = process_mask_obs(
                                            img.copy(), current_mask, mask_pixels=int(height*mask_ratio), scale_mask=np.all(current_mask <= 1), apply_rdp=dataset_config.apply_rdp
                                        )
                                        # frame[f"observation.mask.{cam}"] = masked_img

                                        # Combine path and mask
                                        masked_path_img = process_path_obs(
                                            masked_img.copy(),
                                            current_path,
                                            path_line_size=dataset_config.path_line_size,
                                        )
                                        frame[f"observation.masked_path.{cam}"] = masked_path_img
                            else:
                                frame["camera_present"][cameras.index(cam)] = False
                        frame["camera_present"] = np.array(frame["camera_present"], dtype=bool)
                        if OLD_LEROBOT:
                            dataset.add_frame(frame)
                        else:
                            dataset.add_frame(frame, task=step["language_instruction"].decode())
                    if OLD_LEROBOT:
                        dataset.save_episode(task=step["language_instruction"].decode())
                    else:
                        dataset.save_episode()
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
                    frame["camera_present"] = np.array(frame["camera_present"], dtype=bool) 
                    if not OLD_LEROBOT:
                        frame["task"] = episode["language_instruction"].decode() # new lerobot requires task in frame
                    dataset.add_frame(frame)

                if OLD_LEROBOT:
                    dataset.save_episode(task=step["language_instruction"].decode())
                else:
                    dataset.save_episode()

    # Consolidate the dataset (not needed in lerobot 2.0.0)
    if OLD_LEROBOT:
        dataset.consolidate(run_compute_stats=True)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["widowx", "bridge-v2"],
            private=False,
            push_videos=True,
            license="apache-2.0",
            upload_large_folder=True,
        )


if __name__ == "__main__":
    tyro.cli(main)
