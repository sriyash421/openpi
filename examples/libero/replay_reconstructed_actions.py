"""
Script for replaying reconstructed actions from a model in the Libero environment.

This script:
1. Loads a dataset and extracts ground truth actions
2. Resets the environment to match the dataset's initial state
3. Sends observations and actions to the model for reconstruction
4. Plays the reconstructed actions in the environment
5. Saves a video of the replay
6. Generates a plot showing the difference between GT and reconstructed actions

Usage:
uv run examples/libero/replay_reconstructed_actions.py \
    --dataset_path /path/to/dataset \
    --episode_idx 0 \
    --host 0.0.0.0 \
    --port 8000 \
    --task_suite_name libero_10 \
    --task_id 0 \
    --video_out_path data/libero/replay_videos
"""

import collections
import dataclasses
import logging
import math
import pathlib
from pathlib import Path
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Dataset parameters
    #################################################################################################################
    dataset_path: str  # Path to the LeRobot dataset
    episode_idx: int = 0  # Which episode from the dataset to replay

    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_10"  # Task suite for environment setup
    task_id: int = 0  # Task ID for environment setup
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize

    #################################################################################################################
    # Output parameters
    #################################################################################################################
    video_out_path: str = "data/libero/replay_videos"  # Path to save replay videos
    plot_out_path: str = "data/libero/replay_plots"  # Path to save error plots

    seed: int = 7  # Random seed


def replay_reconstructed_actions(args: Args) -> None:
    """
    Replay reconstructed actions from a model in the Libero environment.
    """
    # Set random seed
    np.random.seed(args.seed)

    # Create output directories
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.plot_out_path).mkdir(parents=True, exist_ok=True)

    # Load dataset
    logging.info(f"Loading dataset from: {args.dataset_path}")
    dataset = LeRobotDataset(args.dataset_path)

    # Get episode info
    if args.episode_idx >= len(dataset.episode_data_index["to"]):
        raise ValueError(
            f"Episode index {args.episode_idx} is out of range. "
            f"Dataset has {len(dataset.episode_data_index['to'])} episodes."
        )

    episode_start = dataset.episode_data_index["from"][args.episode_idx]
    episode_end = dataset.episode_data_index["to"][args.episode_idx]
    episode_length = episode_end - episode_start

    logging.info(f"Episode {args.episode_idx}: frames {episode_start} to {episode_end} ({episode_length} steps)")

    # Get task description from dataset
    task_description = dataset.hf_dataset[episode_start]["task"]
    logging.info(f"Task: {task_description}")

    # Initialize LIBERO environment
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(args.task_id)
    env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

    # Get initial states for this task
    initial_states = task_suite.get_task_init_states(args.task_id)

    # Reset environment to initial state
    env.reset()
    obs = env.set_init_state(initial_states[0])  # Use first initial state

    # Initialize model client
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Storage for replay data
    replay_images = []
    gt_actions = []
    reconstructed_actions = []
    action_errors = []

    logging.info("Starting replay with reconstructed actions...")

    # Wait for objects to stabilize
    for t in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

    # Iterate through episode frames
    for frame_idx in range(episode_start, episode_end):
        relative_idx = frame_idx - episode_start

        # Get ground truth data from dataset
        frame_data = dataset.hf_dataset[frame_idx]
        gt_action = np.array(frame_data["actions"])
        gt_state = np.array(frame_data["state"])

        # Get current observation from environment
        # IMPORTANT: rotate 180 degrees to match train preprocessing
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
        )

        # Save image for replay video
        replay_images.append(img)

        # Prepare input for model reconstruction
        element = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": task_description,
            "actions": gt_action.reshape(1, -1),  # Add batch dimension
            "invert": True,  # Request reconstruction
        }

        # Get reconstructed action from model
        try:
            return_dict = client.infer(element)
            reconstructed_action = return_dict["reconstructed_actions"][0, :7]  # Remove batch dim, take first 7 dims
        except Exception as e:
            logging.error(f"Error getting reconstructed action at step {relative_idx}: {e}")
            reconstructed_action = gt_action[:7]  # Fallback to GT action

        # Calculate error
        error = np.linalg.norm(reconstructed_action - gt_action[:7])
        action_errors.append(error)

        # Store actions for plotting
        gt_actions.append(gt_action[:7])
        reconstructed_actions.append(reconstructed_action)

        # Execute reconstructed action in environment
        obs, reward, done, info = env.step(reconstructed_action.tolist())

        logging.info(f"Step {relative_idx}/{episode_length}: Action error = {error:.6f}")

        if done:
            logging.info(f"Task completed at step {relative_idx}!")
            break

    # Save replay video
    video_filename = f"replay_episode{args.episode_idx}_task{args.task_id}.mp4"
    video_path = pathlib.Path(args.video_out_path) / video_filename
    imageio.mimwrite(
        video_path,
        [np.asarray(x) for x in replay_images],
        fps=10,
    )
    logging.info(f"Replay video saved to: {video_path}")

    # Generate error plot
    _generate_error_plot(
        action_errors,
        gt_actions,
        reconstructed_actions,
        args.plot_out_path,
        args.episode_idx,
        args.task_id,
    )

    # Print statistics
    logging.info("\n" + "=" * 60)
    logging.info("RECONSTRUCTION STATISTICS")
    logging.info("=" * 60)
    logging.info(f"Mean action error: {np.mean(action_errors):.6f}")
    logging.info(f"Std action error: {np.std(action_errors):.6f}")
    logging.info(f"Max action error: {np.max(action_errors):.6f}")
    logging.info(f"Min action error: {np.min(action_errors):.6f}")
    logging.info("=" * 60)


def _generate_error_plot(
    action_errors,
    gt_actions,
    reconstructed_actions,
    plot_out_path,
    episode_idx,
    task_id,
):
    """
    Generate plots showing the difference between GT and reconstructed actions.
    """
    gt_actions = np.array(gt_actions)
    reconstructed_actions = np.array(reconstructed_actions)
    episode_steps = np.arange(len(action_errors))

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Overall L2 error over time
    axes[0].plot(episode_steps, action_errors, linewidth=2, color='red')
    axes[0].set_xlabel('Episode Step')
    axes[0].set_ylabel('L2 Error')
    axes[0].set_title('L2 Error between GT and Reconstructed Actions')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Per-dimension error over time
    per_dim_errors = np.abs(gt_actions - reconstructed_actions)
    for dim in range(per_dim_errors.shape[1]):
        axes[1].plot(episode_steps, per_dim_errors[:, dim], label=f'Dim {dim}', alpha=0.7)
    axes[1].set_xlabel('Episode Step')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title('Per-Dimension Absolute Error')
    axes[1].legend(loc='upper right', ncol=4)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Action values comparison (first 3 dimensions)
    for dim in range(min(3, gt_actions.shape[1])):
        axes[2].plot(episode_steps, gt_actions[:, dim], '--', label=f'GT Dim {dim}', alpha=0.7)
        axes[2].plot(episode_steps, reconstructed_actions[:, dim], '-', label=f'Recon Dim {dim}', alpha=0.7)
    axes[2].set_xlabel('Episode Step')
    axes[2].set_ylabel('Action Value')
    axes[2].set_title('Action Values Comparison (First 3 Dimensions)')
    axes[2].legend(loc='upper right', ncol=3)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_filename = f"error_plot_episode{episode_idx}_task{task_id}.png"
    plot_path = pathlib.Path(plot_out_path) / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    logging.info(f"Error plot saved to: {plot_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(replay_reconstructed_actions)
