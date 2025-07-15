"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:

If you want to push your dataset to the Hugging Face Hub, you can add the `--push_to_hub` flag.

uv run examples/libero/convert_pathmask_libero_data_to_lerobot_vlm_preds.py --data_dir ~/.cache/huggingface/hub/datasets--jesbu1--libero_90_rlds/snapshots/93169e35e1e6ddf6c43171bf038cb4971b60e72a/ \
--paths_masks_file ~/VILA/test_libero_labeling_5x/libero_90_openvla_processed_paths_masks.h5 \
--repo_name jesbu1/libero_90_lerobot_pathmask_vlm_labeled \
--push_to_hub
"""

import shutil
from pathlib import Path
import torch
import torchvision.transforms.functional as F
import cv2

from openpi.policies.mask_path_utils import get_mask_and_path_from_h5
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import numpy as np
import tyro
import os
import json
try:
    # for older lerobot versions before 2.0.0
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    OLD_LEROBOT = True
except ImportError:
    # newer lerobot versions use HF_LEROBOT_HOME instead of LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
    OLD_LEROBOT = False

RAW_DATASET_NAMES = [
    "libero_90_openvla_processed",
    # "libero_10_openvla_processed",
    # "libero_spatial_openvla_processed",
    # "libero_goal_openvla_processed",
    # "libero_object_openvla_processed",
]  # For simplicity we will combine multiple Libero datasets into one training dataset
REPO_NAME = "jesbu1/libero_90_lerobot_pathmask_rdp_vlm_preds"  # Name of the output dataset, also used for the Hugging Face Hub
FLIP_IMAGE = True
DOWNSIZE_IMAGE_SIZE = 224


from vila_utils.utils.decode import add_path_2d_to_img_alt_fast, add_mask_2d_to_img
from vila_utils.utils.encode import scale_path, smooth_path_rdp

def process_path_obs(sample_img, path, path_line_size=3, apply_rdp=False):
    """Process path observation by drawing it onto the image."""
    height, width = sample_img.shape[:2]

    # Scale path to image size
    min_in, max_in = np.zeros(2), np.array([width, height])
    min_out, max_out = np.zeros(2), np.ones(2)
    path_scaled = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)

    if apply_rdp:
        path_scaled = smooth_path_rdp(path_scaled, tolerance=0.05)
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
        mask_points_scaled = smooth_path_rdp(mask_points_scaled, tolerance=0.05)

    return add_mask_2d_to_img(sample_img, mask_points_scaled, mask_pixels=mask_pixels)

def assert_image_valid(img):
    assert img.shape == (256, 256, 3), f"Image shape is {img.shape} but should be (256, 256, 3)"
    assert img.dtype == np.uint8, f"Image dtype is {img.dtype} but should be uint8"
    assert img.max() <= 255, f"Image max is {img.max()} but should be 255"
    assert img.min() >= 0, f"Image min is {img.min()} but should be 0"
    return img

def main(
    data_dir: str,
    paths_masks_file: str = None,  # Make paths_masks_file optional
    path_line_size: int = 2,
    mask_ratio: float = 0.08,
    *,
    push_to_hub: bool = False,
    repo_name: str = REPO_NAME,
):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)


    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=20,
        features={
            "image": {
                "dtype": "video",
                "shape": (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "path_image": {
                "dtype": "video",
                "shape": (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "masked_path_image": {
                "dtype": "video",
                "shape": (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "masked_path_centered_image": {
                "dtype": "video",
                "shape": (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "video",
                "shape": (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
        use_videos=True,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    with h5py.File(paths_masks_file, "r", swmr=True) as path_masks_h5:
        for raw_dataset_name in RAW_DATASET_NAMES:
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode_idx, episode in enumerate(raw_dataset):
                # Initialize path and mask tracking variables
                current_path = None
                current_mask = None
                next_path_timestep_idx = 0
                next_mask_timestep_idx = 0

                #if episode_idx > 10:
                #    break

                for step_idx, step in enumerate(episode["steps"].as_numpy_iterator()):
                    img = step["observation"]["image"]
                    if FLIP_IMAGE:
                        img = np.fliplr(img)
                    frame = {
                        "image": img,
                        "wrist_image": step["observation"]["wrist_image"]
                        if not FLIP_IMAGE
                        else np.fliplr(step["observation"]["wrist_image"]),
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                    }

                    # Get the path and mask data for this episode from HDF5
                    if f"episode_{episode_idx}" in path_masks_h5:
                        episode_group = path_masks_h5[f"episode_{episode_idx}"]
                        # Get path and mask data
                        path_data = episode_group["image_paths"][:] if "image_paths" in episode_group else None
                        path_lengths = episode_group["image_path_lengths"][:] if "image_path_lengths" in episode_group else None
                        mask_data = episode_group["image_masks"][:] if "image_masks" in episode_group else None
                        mask_lengths = episode_group["image_mask_lengths"][:] if "image_mask_lengths" in episode_group else None
                        path_timesteps = episode_group["image_path_timesteps"][:] if "image_path_timesteps" in episode_group else None
                        mask_timesteps = episode_group["image_mask_timesteps"][:] if "image_mask_timesteps" in episode_group else None

                        # Process path and mask if available
                        if path_data is not None:
                            # Check if we need to update the path
                            if step_idx == path_timesteps[next_path_timestep_idx % len(path_timesteps)] and next_path_timestep_idx < len(path_timesteps):
                                current_path = path_data[next_path_timestep_idx, :path_lengths[next_path_timestep_idx]]
                                next_path_timestep_idx += 1

                            # Add path to image if we have one
                            if current_path is not None:
                                path_img = process_path_obs(
                                    img.copy(), current_path, path_line_size=path_line_size, apply_rdp=True
                                )
                                frame["path_image"] = path_img

                                # Process mask if available
                                if mask_data is not None:
                                    # Check if we need to update the mask
                                    if step_idx == mask_timesteps[next_mask_timestep_idx % len(mask_timesteps)] and next_mask_timestep_idx < len(mask_timesteps):
                                        current_mask = mask_data[next_mask_timestep_idx, :mask_lengths[next_mask_timestep_idx]]
                                        next_mask_timestep_idx += 1

                                    # Add mask if we have one
                                    if current_mask is not None:
                                        height, width = step["observation"]["image"].shape[:2]
                                        masked_img = process_mask_obs(
                                            img.copy(),
                                            current_mask,
                                            mask_pixels=int(height * mask_ratio),
                                            scale_mask=np.all(current_mask <= 1),
                                            apply_rdp=True,
                                        )
                                        # Combine path and mask
                                        masked_path_img = process_path_obs(
                                            masked_img.copy(),
                                            current_path,
                                            path_line_size=path_line_size,
                                        )
                                        frame["masked_path_image"] = masked_path_img
                                    else:
                                        frame["masked_path_image"] = img
                                else:
                                    frame["masked_path_image"] = img
                            else:
                                frame["path_image"] = img
                                frame["masked_path_image"] = img
                        else:
                            frame["path_image"] = img
                            frame["masked_path_image"] = img
                    else:
                        frame["path_image"] = img
                        frame["masked_path_image"] = img

                    # center the image around the first point
                    if current_path is not None and len(current_path) > 0:
                        first_point = current_path[0]
                        height, width = frame["masked_path_image"].shape[:2]
                        
                        # Convert first_point to pixel coordinates
                        # Assuming first_point is in normalized coordinates [0, 1]
                        center_x = int(first_point[0] * width)
                        center_y = int(first_point[1] * height)
                        
                        # Calculate crop boundaries
                        crop_size = min(height, width) // 2  # Use half the smaller dimension
                        top = center_y - crop_size
                        left = center_x - crop_size
                        
                            
                        img_tensor = torch.from_numpy(frame["masked_path_image"]).permute(2, 0, 1)
                        cropped_tensor = F.crop(img_tensor, top, left, height, width)
                        frame["masked_path_centered_image"] = cropped_tensor.permute(1, 2, 0).numpy()
                    else:
                        frame["masked_path_centered_image"] = frame["masked_path_image"]

                    if not OLD_LEROBOT:
                        frame["task"] = step["language_instruction"].decode() # new lerobot requires task in frame


                    #downsize all images to 224x224
                    for key in dataset.features:
                        if dataset.features[key]["dtype"] == "video":
                            frame[key] = cv2.resize(frame[key], (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE))

                    dataset.add_frame(frame)
                if OLD_LEROBOT:
                    dataset.save_episode(task=step["language_instruction"].decode())
                else:
                    dataset.save_episode()

    if OLD_LEROBOT:
        # Consolidate the dataset, skip computing stats since we will do that later
        dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            upload_large_folder=True,
            license="apache-2.0",
        )

if __name__ == "__main__":
    tyro.cli(main)
