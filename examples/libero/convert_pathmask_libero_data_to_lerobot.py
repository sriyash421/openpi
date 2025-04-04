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

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import h5py
import numpy as np
import tyro
import os
import json

# TODO: iterate through the dataset_movement_and_masks.h5 file and extract the masks and paths.
RAW_DATASET_NAMES = [
    "libero_90",
    # "libero_10",
    # "libero_spatial",
    # "libero_goal",
    # "libero_object",
]  # For simplicity we will combine multiple Libero datasets into one training dataset
assert len(RAW_DATASET_NAMES) == 1, "Only one dataset name is supported at a time"
REPO_NAME = "jesbu1/libero_90_lerobot_pathmask"  # Name of the output dataset, also used for the Hugging Face Hub


def main(data_dir: str, path_and_mask_file_dir: str, *, push_to_hub: bool = False):
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
                # "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "mask": {
                "dtype": "bool",
                "shape": (256, 256),
                "names": ["height", "width"],
            },
            "path": {
                "dtype": "uint16",
                "shape": (2,),
                "names": ["x", "y"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                # "shape": (128, 128, 3),
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
    with h5py.File(Path(path_and_mask_file_dir) / "dataset_movement_and_masks.h5", "r") as points_and_masks_h5:
        for raw_dataset_name in RAW_DATASET_NAMES:
            # open the directory containing the h5 files using raw_dataset_name
            libero_h5_list = [file for file in os.listdir(Path(data_dir) / raw_dataset_name) if file.endswith(".hdf5")]
            for libero_h5_file in libero_h5_list:
                with h5py.File(Path(data_dir) / raw_dataset_name / libero_h5_file, "r") as f:
                    for demo_num in enumerate(f["data"].keys()):
                        for observation in f["data"][demo_num]["obs"]:
                            gripper_state = observation["gripper_states"]
                            ee_state = observation["ee_states"]
                            state = (np.asarray(np.concatenate((ee_state, gripper_state), axis=-1), np.float32),)
                            # TODO: draw path with opencv2 here, along with the subtrajectory versions.
                            # TODO: save the significant/stopping points? or should we just construct the masks now?
                            # TODO: what else to save?
                            gripper_path = points_and_masks_h5[]
                            dataset.add_frame(
                                {
                                    "image": observation["agentview_image"][
                                        ::-1
                                    ],  # flip the image as it comes from LIBERO reversed
                                    "wrist_image": observation["eye_in_hand_rgb"][
                                        ::-1
                                    ],  # flip the image as it comes from LIBERO reversed
                                    "state": state,
                                    "actions": f["data"][demo_num]["action"],
                                }
                            )
                        # compute language instruction
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
