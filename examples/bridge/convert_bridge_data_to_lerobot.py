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
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None
    # TODO(user): Define image shape expected by LeRobot
    image_height: int = 256
    image_width: int = 256


DEFAULT_DATASET_CONFIG = DatasetConfig()

RAW_DATASET_NAMES = [
    "bridge_v2",
]


def main(
    data_dir: str,
    repo_id: str,
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

    if Path(LEROBOT_HOME / repo_id).exists():
        print(f"Removing existing dataset directory: {LEROBOT_HOME / repo_id}")
        shutil.rmtree(LEROBOT_HOME / repo_id)

    # TODO(user): Set the correct robot_type for WidowX
    robot_type = "widowx"

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=5,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        # video_backend=dataset_config.video_backend,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                frame = {
                    # TODO: check keys
                    "observation.state": step["observation"]["state"],
                    "action": step["action"],
                    "camera_present": [cam in step["observation"] for cam in cameras],
                }
                for cam in cameras:
                    # TODO: how to deal with some cameras not being present? camera_present?
                    if cam in frame:
                        frame[f"observation.images.{cam}"] = step["observation"][cam]

                dataset.add_frame(frame)
            dataset.save_episode(task=step["language_instruction"].decode())

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
