"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_pathmask_libero_data_to_lerobot.py --data_dir /path/to/your/data --path_and_mask_file_dir /path/to/dir_containing_h5points

If you want to push your dataset to the Hugging Face Hub, you can add the `--push_to_hub` flag:

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/jesbu1/libero_openvla_processed_hdf5/
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
from pathlib import Path
import cv2

from openpi.policies.mask_path_utils import get_mask_and_path_from_h5
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
    # "libero_90_openvla_processed",
    "libero_10",
    "libero_spatial",
    "libero_goal",
    "libero_object",
]  # For simplicity we will combine multiple Libero datasets into one training dataset
REPO_NAME = "jesbu1/libero_test_lerobot_pathmask_rdp"  # Name of the output dataset, also used for the Hugging Face Hub
FLIP_IMAGE = True
DOWNSIZE_IMAGE_SIZE = 224


def main(
    data_dir: str,
    path_and_mask_file_dir: str,
    *,
    push_to_hub: bool = False,
    use_subtask_instructions: bool = False,
    return_full_path_mask: bool = False,
    mask_ratio: float = 0.08,
):
    repo_name = REPO_NAME
    if return_full_path_mask:
        repo_name = repo_name + "_full_path_mask"
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
            "masked_image": {
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
    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"-------------------------Processing {raw_dataset_name}-------------------------")
        # open the directory containing the h5 files using raw_dataset_name
        libero_h5_list = [file for file in os.listdir(Path(data_dir) / raw_dataset_name) if file.endswith(".hdf5")]
        for libero_h5_file in libero_h5_list:
            with h5py.File(Path(data_dir) / raw_dataset_name / libero_h5_file, "r", swmr=True) as f:
                for demo_name in f["data"]:
                    num_steps = len(f["data"][demo_name]["obs"]["ee_pos"])
                    try:
                        masked_imgs, path_imgs, masked_path_imgs, quests = get_mask_and_path_from_h5(
                            annotation_path=Path(path_and_mask_file_dir) / "dataset_movement_and_masks.h5",
                            task_key=libero_h5_file.split(".")[0],
                            observation=f["data"][demo_name]["obs"],
                            demo_key=demo_name,
                            return_full_path_mask=return_full_path_mask,
                            mask_ratio=mask_ratio,
                        )
                    except KeyError as e:
                        print(f"KeyError for {demo_name} in {libero_h5_file}: {e}")
                        continue
                    except ValueError as e:
                        print(f"ValueError for {demo_name} in {libero_h5_file}: {e}")
                        continue

                    # Compute the main language instruction
                    if "problem_info" in f["data"].attrs:
                        command = json.loads(f["data"].attrs["problem_info"])["language_instruction"]
                    else:
                        # openvla's language instruction extraction method as it doesn't have problem_info
                        raw_file_string = os.path.basename(libero_h5_file).split("/")[-1]
                        words = raw_file_string[:-10].split("_")
                        command = ""
                        for w in words:
                            if "SCENE" in w:
                                command = ""
                                continue
                            command = command + w + " "
                        command = command[:-1]

                    # Track subtask instructions to divide episodes
                    current_subtask = None

                    assert (
                        len(masked_imgs)
                        == len(path_imgs)
                        == len(masked_path_imgs)
                        == len(quests)
                        == num_steps
                        == len(f["data"][demo_name]["actions"])
                    ), "Lengths of masked_img, path, subtask_path, quests, ee_pos, and action must match"

                    assert masked_imgs[0].max() > 1 and masked_imgs[0].min() == 0, "Masked image must be image"
                    assert path_imgs[0].max() > 1 and path_imgs[0].min() == 0, "Path image must be image"
                    assert (
                        masked_path_imgs[0].max() > 1 and masked_path_imgs[0].min() == 0
                    ), "Masked path image must be image"

                    for i in range(num_steps):
                        gripper_state = f["data"][demo_name]["obs"]["gripper_states"][i]
                        ee_state = f["data"][demo_name]["obs"]["ee_states"][i]
                        state = np.asarray(np.concatenate((ee_state, gripper_state), axis=-1), np.float32)

                        # Get base images and apply flipping if needed
                        agentview_img = f["data"][demo_name]["obs"]["agentview_rgb"][i][
                            ::-1
                        ]  # flip the image as it comes from LIBERO reversed
                        wrist_img = f["data"][demo_name]["obs"]["eye_in_hand_rgb"][i][::-1]

                        if FLIP_IMAGE:
                            agentview_img = np.fliplr(agentview_img)
                            wrist_img = np.fliplr(wrist_img)

                        frame = {
                            "image": agentview_img,
                            "wrist_image": wrist_img,
                            "masked_image": masked_imgs[i],
                            "path_image": path_imgs[i],
                            "masked_path_image": masked_path_imgs[i],
                            "state": state,
                            "actions": f["data"][demo_name]["actions"][i].astype(np.float32),
                        }

                        # Downsize all images to 224x224
                        for key in dataset.features:
                            if dataset.features[key]["dtype"] == "video":
                                frame[key] = cv2.resize(frame[key], (DOWNSIZE_IMAGE_SIZE, DOWNSIZE_IMAGE_SIZE))

                        if not OLD_LEROBOT:
                            # frame["task"] = command
                            dataset.add_frame(frame, task=command)
                        else:
                            dataset.add_frame(frame)

                        # Determine current subtask instruction (if using subtask instructions)
                        if use_subtask_instructions and quests:
                            # Get the subtask for this frame directly
                            new_subtask = quests[i]

                            # If subtask changed or this is the last frame, save the episode
                            if (current_subtask is not None and new_subtask != current_subtask) or i == num_steps - 1:
                                # Save episode with current subtask instruction
                                if current_subtask is None:
                                    if OLD_LEROBOT:
                                        dataset.save_episode(task=command)
                                    else:
                                        dataset.save_episode()
                                else:
                                    if OLD_LEROBOT:
                                        dataset.save_episode(task=current_subtask)
                                    else:
                                        dataset.save_episode()
                                current_subtask = new_subtask
                            # Initialize current_subtask if this is the first frame
                            elif current_subtask is None:
                                current_subtask = new_subtask

                    # If not using subtask instructions, save the entire episode at once
                    if not use_subtask_instructions:
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
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
