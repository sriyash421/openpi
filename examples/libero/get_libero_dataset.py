"""
Script for generating a filtered Libero dataset for a specific task suite and task ID.

This script takes a task suite name and task ID as input, generates a filtered
LeRobot format dataset containing only episodes from that specific task, and saves it.

Usage:
uv run examples/libero/get_libero_dataset.py --data_dir /path/to/your/data --task_suite_name libero_10_no_noops --task_id 0

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/get_libero_dataset.py --data_dir /path/to/your/data --task_suite_name libero_10_no_noops --task_id 0 --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $HF_LEROBOT_HOME directory.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from libero.libero import benchmark
import tensorflow_datasets as tfds
import tyro


AVAILABLE_TASK_SUITES = [
    "libero_10_no_noops",
    "libero_goal_no_noops",
    "libero_object_no_noops",
    "libero_spatial_no_noops",
]

# Map dataset names to benchmark names
DATASET_TO_BENCHMARK_MAP = {
    "libero_10_no_noops": "libero_10",
    "libero_goal_no_noops": "libero_goal",
    "libero_object_no_noops": "libero_object",
    "libero_spatial_no_noops": "libero_spatial",
}


def main(
    data_dir: str,
    task_suite_name: str,
    *,
    # task_id: int = -1,
    language_instruction: str = "",
    push_to_hub: bool = False,
    repo_name_prefix: str = "sriyash421/libero"
):
    """
    Generate a filtered Libero dataset for a specific task suite and task ID.

    Args:
        data_dir: Path to the directory containing raw Libero datasets
        task_suite_name: Name of the task suite (e.g., 'libero_10_no_noops')
        # task_id: ID of the specific task within the suite
        language_instruction: Language instruction to filter episodes
        repo_name_prefix: Prefix for the output dataset repository name
    """
    assert data_dir is not None, "data_dir must be specified"
    assert task_suite_name in AVAILABLE_TASK_SUITES, f"Invalid task_suite_name: {task_suite_name}. Available options: {AVAILABLE_TASK_SUITES}"
    # assert task_id >= 0 or language_instruction != "", "Either task_id or language_instruction must be specified"
    assert language_instruction != "", "language_instruction must be specified"
    # Validate task suite name
    if task_suite_name not in AVAILABLE_TASK_SUITES:
        raise ValueError(
            f"Invalid task_suite_name: {task_suite_name}. "
            f"Available options: {AVAILABLE_TASK_SUITES}"
        )

    # Get the benchmark suite to determine number of tasks
    # benchmark_name = DATASET_TO_BENCHMARK_MAP[task_suite_name]
    # benchmark_dict = benchmark.get_benchmark_dict()
    # task_suite = benchmark_dict[benchmark_name]()
    language_instruction = language_instruction.strip()
    # if language_instruction:
    print(f"Filtering episodes for language instruction: '{language_instruction}'")
    
    num_tasks_in_suite = 10 #task_suite.n_tasks

    # # Validate task_id
    # if task_id < 0 or task_id >= num_tasks_in_suite:
    #     raise ValueError(
    #         f"Invalid task_id: {task_id}. "
    #         f"Task suite '{task_suite_name}' has {num_tasks_in_suite} tasks (valid range: 0-{num_tasks_in_suite - 1})"
    #     )

    # print(f"Task suite '{task_suite_name}' has {num_tasks_in_suite} tasks")

    # Create repository name
    repo_name = f"{repo_name_prefix}_{task_suite_name}_task-{language_instruction}"

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    print(f"Creating dataset: {repo_name}")
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
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

    # Load the raw dataset
    print(f"Loading raw dataset: {task_suite_name} from {data_dir}")
    raw_dataset = tfds.load(task_suite_name, data_dir=data_dir, split="train")

    # Filter episodes for the specific task_id and write to LeRobot dataset
    episode_count = 0
    for episode_idx, episode in enumerate(raw_dataset):
        # Get the task instruction from the first step
        first_step = next(iter(episode["steps"]))
        task_instruction = first_step["language_instruction"].numpy().decode()

        # Check if this episode belongs to the target task_id
        # Episodes are ordered by task_id in the dataset
        episode_task_id = episode_idx % num_tasks_in_suite

        # if episode_task_id == task_id:
        if task_instruction.strip() == language_instruction:
            print(f"Adding episode {episode_count} for task: {task_instruction}")
            for step in episode["steps"].as_numpy_iterator():
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": step["observation"]["state"],
                        "actions": step["action"],
                        "task": step["language_instruction"].decode(),
                    }
                )
            dataset.save_episode()
            episode_count += 1

    print(f"Total episodes added: {episode_count}")

    if episode_count == 0:
        print(f"WARNING: No episodes found for task_id {task_id} in {task_suite_name}")
        print("Please verify that the task_id is valid for this task suite.")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {repo_name}")
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds", task_suite_name, f"task_{task_id}"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_name}")
    else:
        print(f"Dataset saved locally to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
