"""
Script to convert bridge data into a single LeRobot dataset v2.0 format.

Can accept either directories containing task data (inferring trajectories within)
or explicit paths to individual trajectory directories.

Example usage (by task directories):
source .venv/bin/activate
# create a new venv for this script
uv venv
source .venv/bin/activate
uv pip install tensorflow_datasets shapely openai transformers torch torchvision tensorflow opencv-python-headless pillow tyro spacy
uv pip install -e ~/nvidia/my_lerobot
uv pip install -e ~/nvidia/vila_utils
uv run python -m spacy download en_core_web_sm
uv run generate_bridge_sam_masked_lerobot.py --raw-dirs /data/shared/openx_rlds_data/ --repo-id jesbu1/bridge_v2_lerobot_dinosam_masked

Example usage for path masks:
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /data/shared/openx_rlds_data/ --repo_id jesbu1/bridge_v2_lerobot_pathmask --paths-masks-file ~/VILA/test_bridge_labeling_5x/bridge_paths_masks_merged.h5 --push_to_hub
"""

import dataclasses
from pathlib import Path
import shutil
import tensorflow_datasets as tfds
from PIL import Image
import spacy
nlp = spacy.load("en_core_web_sm")

def instruction_to_dino_instr(instruction):
    # find all the nouns in the image and use gripper via spacy
    doc = nlp(instruction)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    nouns = nouns + ["gripper"]

    # add "a " prefix to each object
    objects = ["a " + o for o in nouns]
    # separate objects with ". "
    dino_instr = ". ".join(objects)
    return dino_instr

from grounded_sam_tracker import GroundedSam2Tracker
import tensorflow as tf
import numpy as np
import warnings
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


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    image_writer_processes: int = 0
    image_writer_threads: int = 20
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
    for cam in ["image_0"]:
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
    tracker = GroundedSam2Tracker()
    # Process without paths and masks
    for raw_dataset_name in RAW_DATASET_NAMES:
        print(f"Loading TFDS dataset '{raw_dataset_name}' from {data_dir} (train split)...")
        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")

        episode_iter = iter(raw_dataset)
        episode_count = 0
        processed_episodes = 0
        skipped_episodes = 0
        while True:
            try:
                episode = next(episode_iter)
                episode_idx = episode_count
                episode_count += 1
            except StopIteration:
                break
            except (tf.errors.DataLossError, tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError) as e:
                skipped_episodes += 1
                warnings.warn(f"Skipping corrupted episode at index {episode_count}: {e}")
                continue

            frames_buffer: list[dict] = []
            masks_list = []
            try:
                rgbs = []
                dino_instr = None
                instr = None
                frames=[]
                for step_idx, step in enumerate(episode["steps"].as_numpy_iterator()):
                    # Track language instruction if present
                    if step_idx == 0:
                        if "language_instruction" in step:
                            try:
                                instr = step["language_instruction"].decode()
                            except Exception:
                                pass
                            dino_instr = instruction_to_dino_instr(instr)
                    assert "image_0" in step["observation"]
                    img = step["observation"]["image_0"]
                    frame = {
                        "observation.state": step["observation"]["state"],
                        "action": step["action"],
                        "camera_present": [True],
                    }
                    frames.append(frame)
                    rgbs.append(img)
                assert len(rgbs) > 0

                assert dino_instr is not None
                assert instr is not None

                # now apply the tracker
                masks_list = []
                tracker.reset(init_frame=Image.fromarray(rgbs[0]), text=dino_instr)
                for rgb in rgbs:
                    t, masks = tracker.step(Image.fromarray(rgb))
                    masks_list.append(masks)

                rgbs_masked = tracker.apply_masks_to_frames(rgbs, masks_list)
                for rgb_masked, frame in zip(rgbs_masked, frames):
                    frame[f"observation.images.image_0"] = rgb_masked
                    frames_buffer.append(frame)
                

            except (tf.errors.DataLossError, tf.errors.InvalidArgumentError, tf.errors.OutOfRangeError) as e:
                skipped_episodes += 1
                warnings.warn(f"Skipping episode {episode_idx} due to step read error: {e}")
                frames_buffer = []
                continue

            if len(frames_buffer) == 0:
                continue

            for frame in frames_buffer:
                if not OLD_LEROBOT:
                    frame["task"] = instr
                    dataset.add_frame(frame, task=instr)
                else:
                    dataset.add_frame(frame)

            if OLD_LEROBOT:
                dataset.save_episode(task=instr)
            else:
                dataset.save_episode()
            processed_episodes += 1
            print(f"Saved episode {processed_episodes} (source idx {episode_idx})")
        print(f"Finished TFDS dataset '{raw_dataset_name}'. Episodes processed: {processed_episodes}, skipped: {skipped_episodes}, total seen: {episode_count}.")

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
