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

from openpi.policies.mask_path_utils import get_mask_and_path_from_h5
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import numpy as np
import tyro
import os
import json

RAW_DATASET_NAMES = [
    "libero_90_openvla_processed",
    # "libero_10_openvla_processed",
    # "libero_spatial_openvla_processed",
    # "libero_goal_openvla_processed",
    # "libero_object_openvla_processed",
]  # For simplicity we will combine multiple Libero datasets into one training dataset
assert len(RAW_DATASET_NAMES) == 1, "Only one dataset name is supported at a time"
REPO_NAME = "jesbu1/libero_90_lerobot_pathmask"  # Name of the output dataset, also used for the Hugging Face Hub


def main(
    data_dir: str,
    path_and_mask_file_dir: str,
    *,
    push_to_hub: bool = False,
    use_subtask_instructions: bool = False,
    use_subtask_path: bool = False,
):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "mask": {
                "dtype": "image",
                "shape": (256, 256),
                "names": ["height", "width"],
            },
            "path": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["x", "y"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
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
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        # open the directory containing the h5 files using raw_dataset_name
        libero_h5_list = [file for file in os.listdir(Path(data_dir) / raw_dataset_name) if file.endswith(".hdf5")]
        for libero_h5_file in libero_h5_list:
            with h5py.File(Path(data_dir) / raw_dataset_name / libero_h5_file, "r", swmr=True) as f:
                for demo_name in f["data"]:
                    num_steps = len(f["data"][demo_name]["obs"]["ee_pos"])
                    masks, paths, subtask_paths, quests = get_mask_and_path_from_h5(
                        annotation_path=Path(path_and_mask_file_dir) / "dataset_movement_and_masks.h5",
                        task_key=libero_h5_file.split(".")[0],
                        observation=f["data"][demo_name]["obs"],
                        demo_key=demo_name,
                        hi_start=0,
                        hi_end=num_steps,
                    )

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
                        len(masks)
                        == len(paths)
                        == len(subtask_paths)
                        == len(quests)
                        == num_steps
                        == len(f["data"][demo_name]["actions"])
                    ), "Lengths of mask, path, subtask_path, quests, ee_pos, and action must match"

                    for i in range(num_steps):
                        gripper_state = f["data"][demo_name]["obs"]["gripper_states"][i]
                        ee_state = f["data"][demo_name]["obs"]["ee_states"][i]
                        state = (np.asarray(np.concatenate((ee_state, gripper_state), axis=-1), np.float32),)

                        dataset.add_frame(
                            {
                                "image": f["data"][demo_name]["obs"]["agentview_rgb"][i][
                                    ::-1
                                ],  # flip the image as it comes from LIBERO reversed
                                "wrist_image": f["data"][demo_name]["obs"]["eye_in_hand_rgb"][i][::-1],
                                "mask": masks[i],
                                "path": subtask_paths[i] if use_subtask_path else paths[i],
                                "state": state,
                                "actions": f["data"][demo_name]["actions"][i],
                            }
                        )

                        # Determine current subtask instruction (if using subtask instructions)
                        if use_subtask_instructions and quests:
                            # Get the subtask for this frame directly
                            new_subtask = quests[i]

                            # If subtask changed or this is the last frame, save the episode
                            if (current_subtask is not None and new_subtask != current_subtask) or i == num_steps - 1:
                                # Save episode with current subtask instruction
                                dataset.save_episode(task=current_subtask)
                                current_subtask = new_subtask
                            # Initialize current_subtask if this is the first frame
                            elif current_subtask is None:
                                current_subtask = new_subtask

                    # If not using subtask instructions, save the entire episode at once
                    if not use_subtask_instructions:
                        dataset.save_episode(task=command)

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
