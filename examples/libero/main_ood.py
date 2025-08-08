import collections
import uuid
import shutil
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
from libero.ood.task_distributions import TaskDistribution, AVAILABLE_DISTRIBUTIONS
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
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    draw_path: bool = False
    draw_mask: bool = False

    flip_image_horizontally: bool = True
    mask_ratio: float = 0.1

    distribution_name: str = "distractor_variations" # name of the distribution to use from distractor_variations, visual_variations


    

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

    use_wandb: bool = True  # Whether to also log results in Weights & Biases
    wandb_project: str = "p-masked-vla"  # Name of W&B project to log to (use default!)
    wandb_group_prefix: str = None
    wandb_entity: str = "clvr"  # Name of entity to log under
    wandb_name_suffix: str = ""


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    # set the max steps for the task suite to 500
    max_steps = 500


    if args.use_wandb:
        run_name = f"eval-pi0-{args.distribution_name}_date-{datetime.datetime.now().strftime('%Y-%m-%d')}_seed-{args.seed}_replan-{args.replan_steps}-draw{args.draw_path}-mask{args.draw_mask}{args.mask_ratio}-{args.wandb_name_suffix}"
        if args.wandb_group_prefix:
            group = f"{args.wandb_group_prefix}_vlmfreq{args.vlm_query_frequency}_replan{args.replan_steps}_draw{args.draw_path}_mask{args.draw_mask}{args.mask_ratio}"
        else:
            group = None
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, config=args, group=group)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Find the requested distribution
    distribution = next(
        (dist for dist in AVAILABLE_DISTRIBUTIONS if dist.name == args.distribution_name),
        None,
    )

    # Find all variation pairs (BDDL + XML)
    variations_dir = os.path.join(get_libero_path("benchmark_root"), "../ood", f"test_all_tasks_{distribution.name.split('_')[0]}")
    entries = [
        d for d in os.listdir(variations_dir) if d.startswith(distribution.name + "_")
    ] # list of folders for each variation, one for each task
    variations = []
    for entry in entries:
        task_dir = os.path.join(variations_dir, entry)
    for i in range(distribution.num_variations):
        bddl_file = os.path.join(task_dir, f"variation_{i}.bddl")
        xml_file = os.path.join(task_dir, f"variation_{i}.xml")
        if os.path.exists(bddl_file) and os.path.exists(xml_file):
            variations.append((bddl_file, xml_file))
        else:   
            print(f"No BDDL or XML file found for variation {i} in {task_dir}")

    
    # Start evaluation
    for i, (bddl_file, xml_file) in enumerate(tqdm.tqdm(variations)):
        # Copy XML to assets directory for proper loading
        assets_dir = os.path.join(
            get_libero_path("benchmark_root"),
            "assets",
            "scenes",
        )
        temp_xml_name = f"temp_variation_{uuid.uuid4()}.xml"
        temp_xml = os.path.join(assets_dir, temp_xml_name)
        print(f"  Copying XML to: {temp_xml}")
        shutil.copy2(xml_file, temp_xml)

    
        # Initialize video tracking for this task
        success_videos_saved = 0
        failure_videos_saved = 0

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(bddl_file, temp_xml_name, distribution, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for _ in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

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
                            mask_ratio=args.mask_ratio,
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

            # remove the temp xml file
            if os.path.exists(temp_xml):
                os.remove(temp_xml)

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")

    if args.use_wandb:
        wandb.log({f"{args.task_suite_name}/success_rate": float(total_successes) / float(total_episodes)})
        wandb.finish()

def _get_libero_env(bddl_file, xml_file_name, distribution, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    # get the task description from the folder for the bddl/xml file
    task_description_uncut = os.path.basename(os.path.dirname(bddl_file))
    task_description_cut = task_description_uncut.replace(distribution.name + "_", "") # looks like KITCHEN_SCENE3_turn... or just put_the_bowl...
    if "SCENE" in task_description_cut:
        task_description_cut = task_description_cut.split("SCENE")[1] # looks like 3_turn... or 10_task_name......
        task_description_cut = task_description_cut.split("_")[1] # looks like task name
    task_description = task_description_cut.replace("_", " ")

    # env args
    env_args = {"bddl_file_name": bddl_file, "camera_heights": resolution, "camera_widths": resolution, "scene_xml": f"scenes/{xml_file_name}"}
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
