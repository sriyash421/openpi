"""
Script to convert multiple USC WidowX datasets (potentially with different tasks)
into a single LeRobot dataset v2.0 format.

Example usage: uv run examples/usc_widowx/convert_usc_data_to_lerobot.py --raw-dirs /path/to/task1 /path/to/task2 --repo-id <org>/<combined-dataset-name>
"""

import dataclasses
import re
from pathlib import Path
import shutil
from typing import Literal, Dict, List, Tuple, Optional
import warnings
import pickle

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub._download_raw import download_raw
import numpy as np
import torch
import tqdm
import tyro
from PIL import Image
import torchvision.transforms.functional as F


# TODO(user): If your .dat files are not simple numpy saves, replace this.
def load_data_compressed(filepath: Path) -> Dict[str, np.ndarray]:
    """Placeholder function to load compressed data."""
    try:
        # Assuming .dat files might be numpy archives or pickled objects
        return np.load(filepath, allow_pickle=True).item()  # Use .item() if it's a saved dictionary
    except Exception as e:
        warnings.warn(f"Failed to load {filepath} with numpy.load: {e}. Implement custom loading if needed.")
        raise


def load_pickle_data(filepath: Path) -> Dict[str, np.ndarray]:
    """Load data from a pickle file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


# TODO(user): Adapt this if image loading is different.
def load_images(image_dir: Path) -> np.ndarray:
    """Load images from a directory."""
    image_files = list(image_dir.glob("*.png"))  # Assuming PNG format, adjust if needed
    if not image_files:
        image_files = list(image_dir.glob("*.jpg"))

    # sort the image files by number
    image_files = sorted(
        image_files,
        key=lambda x: int(re.search(r"\d+", x.name).group()),
    )

    if not image_files:
        raise FileNotFoundError(f"No image files (.png, .jpg) found in {image_dir}")

    imgs = []
    for img_file in image_files:
        img = Image.open(img_file).convert("RGB")
        imgs.append(np.array(img))
    return np.stack(imgs)


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001  # Adjust based on data timestamp precision
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None
    # TODO(user): Define image shape expected by LeRobot
    image_height: int = 256
    image_width: int = 256


DEFAULT_DATASET_CONFIG = DatasetConfig()


def get_trajectory_paths(raw_dirs: List[Path]) -> List[Tuple[Path, str]]:
    """Find all trajectory directories within the list of raw data directories
       and associate them with a task name derived from the parent directory."""
    all_traj_infos = []
    for raw_dir in raw_dirs:
        if not raw_dir.is_dir():
            warnings.warn(f"Provided raw directory path is not a directory, skipping: {raw_dir}")
            continue

        # Assuming the task name is the name of the directory containing traj folders
        task_name = raw_dir.name
        task_name_processed = task_name.replace("_", " ").capitalize() # Process for better readability
        print(f"Processing task: {task_name_processed}")

        # Find trajectory folders (e.g., traj0, traj1, ...) within this raw_dir
        traj_paths_in_dir = [p for p in raw_dir.iterdir() if p.is_dir() and re.match(r"traj\d+", p.name)]

        if not traj_paths_in_dir:
            warnings.warn(f"No trajectory subdirectories found in {raw_dir}")
            continue

        print(f"Found {len(traj_paths_in_dir)} potential trajectory directories in {raw_dir} for task '{task_name_processed}'.")
        for traj_path in traj_paths_in_dir:
            all_traj_infos.append((traj_path, task_name_processed))

    if not all_traj_infos:
        raise FileNotFoundError(f"No trajectory subdirectories found across all provided raw directories: {raw_dirs}")

    print(f"Found a total of {len(all_traj_infos)} trajectories across all directories.")
    return all_traj_infos


def create_empty_dataset(
    repo_id: str,
    mode: Literal["video", "image"] = "video",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    # TODO(user): Verify motor names and count for WidowX.
    state = [
        # Example names, replace with actual motor names used in your data
        "x",
        "y",
        "z",
        "x_angle",
        "y_angle",
        "z_angle",
        "gripper",
    ]
    cameras = [
        "external",
        "over_shoulder",
        # Add other camera names if present, e.g., "wrist"
    ]

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
            "shape": (len(state),),
            "names": state,
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, dataset_config.image_height, dataset_config.image_width),
            "names": [
                "channels",
                "height",
                "width",
            ],
        }

    if Path(LEROBOT_HOME / repo_id).exists():
        print(f"Removing existing dataset directory: {LEROBOT_HOME / repo_id}")
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # TODO(user): Set the correct robot_type for WidowX
    robot_type = "widowx"

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_raw_episode_data(
    traj_path: Path,
    cameras: List[str],
    dataset_config: DatasetConfig,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Load state, action, and image data for a single trajectory."""
    obs_file = traj_path / "obs_dict.pkl"
    action_file = traj_path / "policy_out.pkl"

    if not obs_file.exists():
        raise FileNotFoundError(f"Observation file not found: {obs_file}")
    if not action_file.exists():
        raise FileNotFoundError(f"Action file not found: {action_file}")

    obs_data = load_pickle_data(obs_file)
    action_data = load_pickle_data(action_file)

    assert isinstance(action_data, list)
    assert isinstance(obs_data, dict)

    # Ensure data is numpy before converting to tensor
    state_np = obs_data["state"]
    action_np = np.concatenate([step["actions"][None] for step in action_data], axis=0)
    if not isinstance(state_np, np.ndarray):
        raise TypeError(f"Expected 'state' in {obs_file} to be numpy array, got {type(state_np)}")
    if not isinstance(action_np, np.ndarray):
        raise TypeError(f"Expected 'actions' in {action_file} to be numpy array, got {type(action_np)}")

    state = torch.from_numpy(state_np)  # Key based on TFDS script
    action = torch.from_numpy(action_np)  # Key based on TFDS script

    # Check lengths
    num_frames = state.shape[0]
    if action.shape[0] != num_frames:
        warnings.warn(
            f"State ({num_frames}) and action ({action.shape[0]}) length mismatch in {traj_path}. Truncating to minimum."
        )
        min_len = min(num_frames, action.shape[0])
        state = state[:min_len]
        action = action[:min_len]
        num_frames = min_len

    imgs_per_cam = {}
    for camera in cameras:
        # Assuming images are in a subdirectory named after the camera.
        image_dir = traj_path / f"{camera}_imgs"
        imgs_np = load_images(image_dir)
        print(f"Loaded {imgs_np.shape[0]} images from directory {image_dir}")

        # Preprocess images: center crop and resize
        if imgs_np.size == 0:
            warnings.warn(f"Empty image array loaded from {image_dir}. Skipping camera {camera}.")
            continue

        n, h, w, c = imgs_np.shape
        if c != 3:
            warnings.warn(f"Expected 3 channels, got {c} in {image_dir}. Skipping camera {camera}.")
            continue

        # Convert to tensor (N, C, H, W), normalize to [0, 1]
        imgs_tensor = torch.from_numpy(imgs_np).permute(0, 3, 1, 2).float() / 255.0

        # Center crop to square
        crop_size = min(h, w)
        cropped_tensor = F.center_crop(imgs_tensor, output_size=[crop_size, crop_size])

        # Resize to target dimensions
        resized_tensor = F.resize(
            cropped_tensor, size=[dataset_config.image_height, dataset_config.image_width], antialias=True
        )

        # Store the processed tensor (N, C, H_out, W_out)
        imgs_per_cam[camera] = resized_tensor


        # Verify image count
        if camera in imgs_per_cam and imgs_per_cam[camera].shape[0] != num_frames:
            # if it's 1 more than the number of frames, just remove the last frame
            if imgs_per_cam[camera].shape[0] == num_frames + 1:
                imgs_per_cam[camera] = imgs_per_cam[camera][:-1]
            else:
                warnings.warn(
                    f"Image count ({imgs_per_cam[camera].shape[0]}) for camera {camera} does not match state/action count ({num_frames}) in {traj_path}. Skipping camera."
                )
                del imgs_per_cam[camera]

    # Return only cameras that were successfully loaded and matched frame count
    valid_cameras = list(imgs_per_cam.keys())
    if not valid_cameras:
        # If no valid cameras remain (e.g., all had frame count mismatches), return None for images
        warnings.warn(f"No valid image data found for any camera in {traj_path}. Returning None for images.")
        return None, state, action # Return state/action in case they are still useful without images

    # Ensure all required cameras are present after validation
    missing_required = [cam for cam in cameras if cam not in valid_cameras]
    if missing_required:
         warnings.warn(f"Required cameras {missing_required} missing valid data in {traj_path}. Returning None for images.")
         return None, state, action


    return imgs_per_cam, state, action


def populate_dataset(
    dataset: LeRobotDataset,
    traj_infos: List[Tuple[Path, str]], # List of (trajectory_path, task_name)
    dataset_config: DatasetConfig,
    episodes: Optional[List[int]] = None,
) -> LeRobotDataset:
    """Populate the LeRobotDataset with data from trajectory files."""
    if episodes is None:
        # If specific episodes are requested, filter traj_infos
        # Note: This applies indices across *all* found trajectories, not per task/directory
        selected_traj_infos = [traj_infos[i] for i in range(len(traj_infos))]
    else:
        if any(i >= len(traj_infos) for i in episodes):
             raise IndexError(f"Episode index out of bounds. Requested indices {episodes}, but found {len(traj_infos)} total trajectories.")
        selected_traj_infos = [traj_infos[i] for i in episodes]


    # Get camera names from dataset features
    cameras = [key.split(".")[-1] for key in dataset.features if key.startswith("observation.images.")]

    num_added_episodes = 0
    # Iterate through the selected (trajectory path, task name) tuples
    for ep_idx, (traj_path, task) in enumerate(tqdm.tqdm(selected_traj_infos, desc="Processing trajectories")):
        print(f"\nProcessing trajectory: {traj_path.name} (Task: {task})")
        try:
            loaded_data = load_raw_episode_data(traj_path, cameras, dataset_config)
            # Check if image loading failed
            if loaded_data[0] is None:
                 warnings.warn(f"Skipping trajectory {traj_path.name} due to missing/invalid image data.")
                 continue
            imgs_per_cam, state, action = loaded_data

        except FileNotFoundError as e:
            warnings.warn(f"Skipping trajectory {traj_path.name}: {e}")
            continue
        except TypeError as e:
             warnings.warn(f"Skipping trajectory {traj_path.name} due to data type error: {e}")
             continue
        except Exception as e:
             warnings.warn(f"Skipping trajectory {traj_path.name} due to unexpected error: {e}")
             continue


        num_frames = state.shape[0]

        if num_frames == 0:
            warnings.warn(f"Skipping empty trajectory: {traj_path.name}")
            continue
        # Basic check (more robust checks happen in load_raw_episode_data)
        if not imgs_per_cam or cameras[0] not in imgs_per_cam or num_frames != len(imgs_per_cam[cameras[0]]):
             warnings.warn(f"Frame count mismatch or missing camera data for {traj_path.name}. State: {num_frames}, Action: {action.shape[0]}, Images: {len(imgs_per_cam.get(cameras[0], []))}. Skipping.")
             continue


        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
            }

            all_cams_present = True
            for camera in cameras:
                if camera not in imgs_per_cam:
                    warnings.warn(
                        f"Camera {camera} missing image data for frame {i} in {traj_path.name}. Skipping frame."
                    )
                    all_cams_present = False
                    break  # Skip this frame if any required camera is missing
                # Ensure image is in CHW format for LeRobotDataset
                img = imgs_per_cam[camera][i]  # This is now a CHW tensor

                # Assign the CHW tensor directly
                frame[f"observation.images.{camera}"] = img

            assert all_cams_present, f"Camera {camera} missing image data for frame {i} in {traj_path.name}. Skipping frame."

            dataset.add_frame(frame)

        dataset.save_episode(task=task)
        num_added_episodes += 1
        print(f"Saved episode {num_added_episodes} from {traj_path.name} with {num_frames} frames.")

    print(f"Finished processing. Added {num_added_episodes} episodes.")
    return dataset


def port_usc_data(
    raw_dirs: List[Path], # Changed from raw_dir: Path
    repo_id: str,
    # task: str = "default_task", # Removed, inferred from dir name
    raw_repo_id: Optional[str] = None, # Keep this for potential single download source? Maybe remove. Let's keep for now.
    episodes: Optional[List[int]] = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "video",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    """Main function to convert USC data from multiple directories and optionally push to Hub."""
    if (LEROBOT_HOME / repo_id).exists():
        print(f"Dataset already exists locally at {LEROBOT_HOME / repo_id}. Removing.")
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # Check if *any* of the raw_dirs exist. Downloading is complex with multiple dirs.
    # Let's require directories to exist locally for now.
    # TODO(user): Add support for downloading multiple raw_repo_ids if needed.
    existing_raw_dirs = [d for d in raw_dirs if d.exists()]
    if not existing_raw_dirs:
         # If raw_repo_id is provided, attempt download to the *first* path in raw_dirs
         # This assumes raw_repo_id contains all necessary data, which might not be true
         # for the multi-directory case. Recommend pre-downloading for multi-dir.
        if raw_repo_id:
             target_download_dir = raw_dirs[0]
             warnings.warn(f"None of the specified raw directories exist. Attempting to download from {raw_repo_id} into {target_download_dir}. This might not contain all required task data.")
             print(f"Downloading from {raw_repo_id} to {target_download_dir}...")
             # Create parent if it doesn't exist
             target_download_dir.parent.mkdir(parents=True, exist_ok=True)
             download_raw(target_download_dir, repo_id=raw_repo_id)
             # Re-check existence after download
             existing_raw_dirs = [d for d in raw_dirs if d.exists()]
             if not existing_raw_dirs:
                  raise FileNotFoundError(f"Raw data directory {target_download_dir} still not found after attempting download from {raw_repo_id}.")
        else:
             raise FileNotFoundError(f"None of the specified raw directories exist: {raw_dirs}, and no raw_repo_id provided for download.")

    # Warn if some directories were provided but don't exist
    missing_dirs = [d for d in raw_dirs if not d.exists()]
    if missing_dirs:
        warnings.warn(f"The following raw directories do not exist and will be skipped: {missing_dirs}")


    # Use only the existing directories
    traj_infos = get_trajectory_paths(existing_raw_dirs)
    if not traj_infos:
        print(f"No valid trajectory paths found in the existing directories: {existing_raw_dirs}. Exiting.")
        return

    dataset = create_empty_dataset(
        repo_id,
        mode=mode,
        dataset_config=dataset_config,
    )
    dataset = populate_dataset(
        dataset,
        traj_infos, # Pass the list of (path, task) tuples
        # task=task, # Removed
        dataset_config=dataset_config,
        episodes=episodes,
    )

    if dataset.num_episodes > 0:
        print("Consolidating dataset...")
        dataset.consolidate()

        if push_to_hub:
            print("Pushing dataset to Hugging Face Hub...")
            dataset.push_to_hub()
            print(f"Successfully pushed {repo_id} to Hub.")
        else:
            print(f"Dataset saved locally at {LEROBOT_HOME / repo_id}. Skipping push to Hub.")
    else:
        print("No episodes were successfully added to the dataset.")


if __name__ == "__main__":
    tyro.cli(port_usc_data)
