"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:

If you want to push your dataset to the Hugging Face Hub, you can add the `--push_to_hub` flag.

or ~/VILA/libero_90_labels_3b/libero_90_openvla_processed_paths_masks.h5

uv pip install tensorflow_datasets 

uv run examples/fractal/convert_fractal_data_to_lerobot.py \
--data_dir /data/shared/openx_rlds_data/ \
--paths_masks_file ~/VILA/fractal_labels_3b/fractal20220817_data_paths_masks.h5 \
--repo_name jesbu1/fractal_lerobot_pathmask_vlm_labeled \
--push_to_hub
"""

import shutil
import tensorflow_datasets as tfds
import tensorflow as tf
import cv2
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import numpy as np
import tyro
try:
    # for older lerobot versions before 2.0.0
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    OLD_LEROBOT = True
except ImportError:
    # newer lerobot versions use HF_LEROBOT_HOME instead of LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
    OLD_LEROBOT = False

RAW_DATASET_NAMES = [
    "fractal20220817_data",
]  
REPO_NAME = "jesbu1/fractal_lerobot_pathmask_vlm_labeled"  # Name of the output dataset, also used for the Hugging Face Hub
DOWNSIZE_IMAGE_SIZE = 224
DEBUG = True


from vila_utils.utils.decode import add_path_2d_to_img_alt_fast, add_mask_2d_to_img
from vila_utils.utils.encode import scale_path, smooth_path_rdp

def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions

def rt1_action_extraction(step: dict) -> dict: # from OpenVLA's OXE Transform for RT1
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = step["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    action = tf.concat(
        (
            step["action"]["world_vector"],
            step["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    return action

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
    mask_ratio_min: float = 0.05,
    mask_ratio_max: float = 0.10,
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
        fps=5,
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
            # Load dataset builder
            builder = tfds.builder(raw_dataset_name, data_dir=data_dir)
            # Define custom decoding behavior for the field that expects 256 x 320 for some reason when
            # it should be 256 x 256
            custom_decoder = {
                "steps": {
                    "observation": {
                        "image": tfds.decode.SkipDecoding(),  # We'll decode manually later
                    }
                }
            }

            # Load the dataset, skipping strict shape checks
            raw_dataset = builder.as_dataset(split="train", decoders=custom_decoder)
            #raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            for episode_idx, episode in enumerate(raw_dataset):
                if DEBUG:
                    if episode_idx > 20:
                        break
                mask_ratio = np.random.uniform(
                    mask_ratio_min, mask_ratio_max
                )  # set the masking ratio for this entire episode
                # Initialize path and mask tracking variables
                current_path = None
                current_mask = None
                next_path_timestep_idx = 0
                next_mask_timestep_idx = 0

                for step_idx, step in enumerate(episode["steps"]):
                    img = tf.io.decode_image(step["observation"]["image"]).numpy()
                    state = tf.concat([step["observation"]["base_pose_tool_reached"], step["observation"]["gripper_closed"]], axis=-1).numpy()
                    action = rt1_action_extraction(step).numpy()
                    frame = {
                        "image": img,
                        "state": state,
                        "actions": action,
                    }
                    # Get task description for this step
                    command = (
                        step["observation"]["natural_language_instruction"]
                        .numpy()
                        .decode()
                    )

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

                    #downsize all images to 224x224
                    for key in dataset.features:
                        if dataset.features[key]["dtype"] == "video":
                            frame[key] = cv2.resize(frame[key], (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE))

                    if OLD_LEROBOT:
                        dataset.add_frame(frame)
                    else:
                        dataset.add_frame(frame, task=command)
                if OLD_LEROBOT:
                    dataset.save_episode(task=command)
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
