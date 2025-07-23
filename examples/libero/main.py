import collections
import datetime
import wandb
import dataclasses
import logging
import math
import pathlib
from pathlib import Path
from typing import Union

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from vila_utils.utils.encode import scale_path
import tqdm
import tyro
import sys
import os
import h5py

from src.openpi.policies.eval_maskpath_utils import get_path_mask_from_vlm

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    vlm_server_ip: str = "http://0.0.0.0:8000"
    vlm_query_frequency: int = 5  # call VLM once every how many action chunks

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    draw_path: bool = False
    draw_mask: bool = False

    flip_image_horizontally: bool = True
    mask_ratio: float = 0.08

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    use_wandb: bool = True  # Whether to also log results in Weights & Biases
    wandb_project: str = "p-masked-vla"  # Name of W&B project to log to (use default!)
    wandb_entity: str = "clvr"  # Name of entity to log under
    wandb_name_suffix: str = ""


def _load_path_and_mask_from_h5(
    path_and_mask_h5_file: Path,
    task_description: str,
    episode_idx: int,
    img_shape: tuple,
):
    """Load path and mask data from HDF5 file.

    Args:
        path_and_mask_h5_file: Path to the HDF5 file containing path and mask data
        task_description: Description of the task
        episode_idx: Index of the episode
        img_shape: Shape of the image for scaling path and mask

    Returns:
        Tuple of (path, mask) where each can be None if loading fails
    """
    try:
        with h5py.File(path_and_mask_h5_file, "r", swmr=True) as f:
            # Find a key that contains the task description
            task_key = None
            task_description_clean = task_description.replace(" ", "_")
            for key in f.keys():
                if task_description_clean in key:
                    task_key = key
                    break

            if task_key is None:
                raise KeyError(f"Could not find task key containing '{task_description_clean}' in HDF5 file")

            demo_key = f"demo_{episode_idx}"
            f_annotation = f[task_key][demo_key]["primary"]

            # Get path data
            path = f_annotation["gripper_positions"]  # Get path

            # Get mask data
            significant_points = f_annotation["significant_points"][0]
            stopped_points = f_annotation["stopped_points"][0]
            mask = np.concatenate([significant_points, stopped_points], axis=0)

            # Scale path and mask to 0, 1-normalized coordinates for VLM to scale back to image coords.
            w, h = img_shape[:2]
            min_in, max_in = np.zeros(2), np.array([w, h])
            min_out, max_out = np.zeros(2), np.ones(2)
            path = scale_path(path, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)
            mask = scale_path(mask, min_in=min_in, max_in=max_in, min_out=min_out, max_out=max_out)

            return path, mask
    except (KeyError, ValueError) as e:
        logging.warning(f"Failed to load ground truth path and mask: {e}, skipping")
        return None, None


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    if args.use_wandb:
        run_name = f"pi0-{args.task_suite_name}_date-{datetime.datetime.now().strftime('%Y-%m-%d')}_seed-{args.seed}_replan-{args.replan_steps}-draw{args.draw_path}-mask{args.draw_mask}-{args.wandb_name_suffix}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)
        task_description = task.language

        # Initialize video tracking for this task
        success_videos_saved = 0
        failure_videos_saved = 0

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []

            # initialize vlm path and and query counter
            path = None
            mask = None
            vlm_query_counter = 0

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
                    if (
                        args.flip_image_horizontally
                    ):  # for models trained with the original OpenVLA processed data, not the pathmask new data
                        img = img[:, ::-1]
                        wrist_img = wrist_img[:, ::-1]
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk

                        # get path and mask from VLM
                        if args.draw_path or args.draw_mask:
                            # Use VLM to get path and mask
                            if vlm_query_counter % args.vlm_query_frequency == 0:
                                vlm_query_counter = 0
                                # setting path and mask to None so that the VLM is called
                                path, mask = None, None
                            img, path, mask = get_path_mask_from_vlm(
                                img,
                                "Center Crop",
                                str(task_description),
                                draw_path=args.draw_path,
                                draw_mask=args.draw_mask,
                                verbose=True,
                                vlm_server_ip=args.vlm_server_ip,
                                path=path,
                                mask=mask,
                                mask_ratio=args.mask_ratio,
                            )
                            vlm_query_counter += 1
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )

                        # Prepare observations dict
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
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])
                    elif args.draw_path or args.draw_mask:
                        # draw path and mask on image just for visualization when action chunk is still being used
                        # Use VLM to get path and mask
                        img, path, mask = get_path_mask_from_vlm(
                            img,
                            "Center Crop",
                            str(task_description),
                            draw_path=args.draw_path,
                            draw_mask=args.draw_mask,
                            verbose=True,
                            vlm_server_ip=args.vlm_server_ip,
                            path=path,
                            mask=mask,
                        )
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )

                    action = action_plan.popleft()

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4"
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log video to wandb if we haven't reached the limit for this category
            if args.use_wandb:
                if done and success_videos_saved <= 2:
                    success_videos_saved += 1
                    wandb.log({
                        f"videos/{task_description}/success_{success_videos_saved}": wandb.Video(str(video_path), fps=10)
                    })
                elif not done and failure_videos_saved <= 2:
                    failure_videos_saved += 1
                    wandb.log({
                        f"videos/{task_description}/failure_{failure_videos_saved}": wandb.Video(str(video_path), fps=10)
                    })

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    if args.use_wandb:
        wandb.log({f"{args.task_suite_name}/success_rate": float(total_successes) / float(total_episodes)})
        wandb.finish()

def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
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
    tyro.cli(eval_libero)
