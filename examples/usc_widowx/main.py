#!/usr/bin/env python
"""Inference script for USC WidowX using openpi policy server.

Example usage:
python examples/usc_widowx/main.py --policy-server-address https://whippet-pet-singularly.ngrok.app --robot-ip localhost --robot-port 5556 --prompt "pick up the red block"
"""

import argparse
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np
from pynput import keyboard

# --- openpi specific imports ---
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client import image_tools

# --- widowx specific imports (assuming installed from widowx_envs or similar) ---
# Need to ensure these imports are correct based on the actual widowx_envs structure
try:
    from widowx_envs.widowx_env_service import WidowXClient, WidowXConfigs, show_video
    from widowx_envs.utils.exceptions import Environment_Exception
    from widowx_envs.utils.raw_saver import RawSaver # For saving trajectories
except ImportError as e:
    print(f"Error importing widowx_envs components: {e}")
    print("Please ensure widowx_envs is installed correctly.")
    exit(1)

resolution = 256

# --- Globals for keyboard listener ---
key_pressed = None

class WidowXConfigs:
    DefaultEnvParams = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, -0.1, -1.57, 0],
            [0.45, 0.25, 0.25, 1.57, 0],
        ],
        "catch_environment_except": False,
        "start_state": [
            0.11865137,
            -0.01696823,
            0.24405071,
            -0.03702571,
            -0.11837727,
            0.03907566,
            0.9994886,
        ],
        "skip_move_to_neutral": False,
        "return_full_image": False,
        "camera_topics": [
            {"name": "/D435/color/image_raw"},
            {"name": "/blue/image_raw"},
            # {"name": "/yellow/image_raw"},
        ],
    }

camera_to_name_map = {
    "external_img": "external",
    "over_shoulder_img": "over_shoulder",
}


def on_press(key):
    """Callback for key press events"""
    global key_pressed
    try:
        if key.char.lower() in ["r", "s"]:
            key_pressed = key.char.lower()
    except AttributeError:
        pass


def on_release(key):
    """Callback for key release events"""
    global key_pressed
    key_pressed = None


def check_key_press():
    """
    Check for 'R' or 'S' key press without blocking.
    Returns: 'r', 's', or None
    """
    global key_pressed
    return key_pressed


def start_keyboard_listener():
    print("Starting keyboard listener: Press 'q' to stop, 'r' to reset.")
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener

def init_robot(robot_ip: str, robot_port: int = 5556) -> WidowXClient:
    """Initializes connection to the WidowX robot."""
    print(f"Connecting to WidowX controller @ {robot_ip}...")
    # Adjust DefaultEnvParams if necessary for your setup
    # Example modification: set specific cameras if needed by controller init
    # env_params['camera_topics'] = [...]
    widowx_client = WidowXClient(host=robot_ip, port=robot_port)
    widowx_client.init(WidowXConfigs.DefaultEnvParams, image_size=resolution)
    print("Successfully connected to WidowX.")
    print("Waiting for initial observation...")
    wait_for_observation(widowx_client)
    print("Initial observation received.")
    print("Resetting robot...")
    widowx_client.reset()
    print("Robot reset.")
    print("Showing video...")
    show_video(widowx_client, duration=2.5)
    print("Video shown. Robot ready")
    return widowx_client

def wait_for_observation(client: WidowXClient, timeout: int = 60) -> Dict:
    """Wait for and return a valid observation from the robot."""
    start_time = time.time()
    while True:
        obs = client.get_observation()
        if obs is not None:
            print("✓ Received valid observation from robot")
            return obs

        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"No observation received from robot after {timeout}s")

        time.sleep(1)
        print(f"⏳ Waiting for robot observation... (elapsed: {elapsed:.1f}s)")


def format_observation(raw_obs: Dict[str, Any], cameras: List[str], prompt: str) -> Dict[str, Any]:
    """Formats raw observation from robot into the structure expected by the policy."""
    obs_for_policy = {
        "state": raw_obs["state"],
        "prompt": prompt,
    }
    for cam_name in cameras:
        # Map camera name to the key used in raw_obs
        assert (
            f"{cam_name}_img" in raw_obs
        ), f"Camera image key '{cam_name}_img' not found in raw observation. Available keys: {raw_obs.keys()}"
        img_key = f"{cam_name}_img"

        img = raw_obs[img_key]

        # Policy expects keys like 'external', 'over_shoulder' directly under 'images'
        obs_for_policy[f"images/{cam_name}"] = image_tools.resize_with_pad(img, resolution, resolution)

    return obs_for_policy

def run_inference_loop(
    args: argparse.Namespace,
    policy_client: _websocket_client_policy.WebsocketClientPolicy,
    widowx_client: WidowXClient,
    saver: RawSaver,
    episode_idx: int,
) -> bool:
    """Runs the main observation-inference-action loop.

    Returns:
        bool: True if reset was requested, False otherwise (stop or normal finish).
    """
    global key_pressed
    key_pressed = None

    listener = start_keyboard_listener()

    raw_obs_list = [] # Store raw obs for saving
    action_list = [] # Store received action chunks

    try:
        print("Resetting robot to start position...")
        # Use start_obs=True if your reset function returns the first observation
        # obs_info = widowx_client.reset(start_obs=True) 
        # raw_obs = obs_info[0] if obs_info else None # Adapt based on reset return value
        
        # If reset doesn't return obs, call get_observation separately
        widowx_client.reset()
        time.sleep(1.0)  # Allow time for reset
        raw_obs = wait_for_observation(widowx_client)
        
        if raw_obs is None:
            print("Failed to get initial observation. Exiting rollout.")
            return False # Indicate failure -> stop

        print("Initial observation received.")
        num_steps = 0
        start_time = time.time()

        while True: # Loop until stop, reset, or error
            loop_start_time = time.time()
            
            # Check keyboard flags first
            if key_pressed == "s":
                 break
            # Check for key press
            key = check_key_press()
            if key == "r":
                print("\nReset requested by user")
                widowx_client.reset()
                wait_for_observation(widowx_client)
                return False, "Reset requested by user"
            elif key == "s":
                print("\nSave and continue requested by user")
                return True, "Saved mid-trajectory by user"

            # 1. Format observation for policy
            try:
                obs_for_policy = format_observation(raw_obs, args.cameras, args.prompt)
            except ValueError as e:
                print(f"Error formatting observation: {e}. Stopping rollout.")
                return False, "Error formatting observation"

            # 2. Get action from policy server
            action_chunk = None
            try:
                inference_start_time = time.time()
                result = policy_client.infer(obs_for_policy)
                inference_time = time.time() - inference_start_time
                # Server returns list of lists, convert to numpy array [chunk_len, action_dim]
                action_chunk = np.array(result["actions"]) 
                if action_chunk.ndim != 2 or action_chunk.shape[1] != 7: # Basic validation
                    print(f"Warning: Unexpected action chunk shape received: {action_chunk.shape}")
                    # Decide how to handle - skip step? stop? For now, continue but log.
            except Exception as e:
                print(f"Error during inference: {e}. Stopping rollout.")
                return False, "Error during inference"
            for i, action in enumerate(action_chunk):
                if i == args.max_action_length - 1:
                    break
                # Store raw observation and received action chunk *before* execution
                raw_obs_list.append(raw_obs)

                # 3. Execute the first action in the chunk
                action_to_execute = action
                # step_action returns next_obs, reward, done, info - we only need next_obs
                step_result = widowx_client.step_action(action_to_execute)

                num_steps += 1

                # 4. Maintain control frequency
                loop_time = time.time() - loop_start_time
                sleep_time = (1.0 / args.hz) - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # Log if we are falling behind
                    if num_steps % 10 == 0:
                        print(
                            f"Warning: Loop running slower than {args.hz} Hz. Target: {1.0/args.hz:.4f}s, Actual: {loop_time:.4f}s, Inference: {inference_time:.4f}s"
                        )
            raw_obs = wait_for_observation(widowx_client)

            # --- End of loop --- #

        rollout_time = time.time() - start_time
        print(f"Rollout ended. Steps: {num_steps}, Duration: {rollout_time:.2f}s")

        # Save data if the loop finished or was stopped (but not reset)
        save_trajectory(saver, episode_idx, raw_obs_list, action_list)
        return False # Indicate stop or normal finish

    except Exception as e:
         print(f"An error occurred during the inference loop: {e}")
         # Attempt to save any data collected before the error
         save_trajectory(saver, episode_idx, raw_obs_list, action_list, success=False, notes=f"Error: {str(e)}")
         return False # Indicate abnormal stop
    finally:
        print("Stopping keyboard listener.")
        listener.stop()
        # Ensure listener thread joins
        listener.join()

def save_trajectory(saver: RawSaver, episode_idx: int, raw_obs_list: List[Dict], action_list: List[np.ndarray]):
    """Saves the collected trajectory data using RawSaver."""
    if not raw_obs_list or not action_list:
        print("No data collected, skipping save.")
        return

    print(f"Saving trajectory {episode_idx}...")
    try:
        # Create obs_dict: keys are observation names, values are lists/arrays over time
        obs_dict = {}
        if raw_obs_list:
            first_obs = raw_obs_list[0]
            for key in first_obs:
                # Stack arrays if possible (like images, state), otherwise keep as list
                try:
                    # Check if all elements are numpy arrays before stacking
                    if all(isinstance(obs.get(key), np.ndarray) for obs in raw_obs_list):
                         stacked_data = np.stack([obs[key] for obs in raw_obs_list])
                         obs_dict[key] = stacked_data
                    else:
                         # Keep as list if types are mixed or not stackable
                         obs_dict[key] = [obs.get(key) for obs in raw_obs_list]
                except Exception as stack_err:
                    print(f"  - Could not stack key '{key}', saving as list. Error: {stack_err}")
                    obs_dict[key] = [obs.get(key) for obs in raw_obs_list]


        # Create agent_data: includes actions, success marker, and notes
        # RawSaver expects actions for N steps. Our loop collects N action chunks.
        # Let's save the first action from each chunk, as that's what was executed.
        executed_actions = np.array([chunk[0] for chunk in action_list if chunk.ndim >= 2 and chunk.shape[0] > 0])

        agent_data = {
            "actions": executed_actions,
            # Optionally save the full action chunks if needed for analysis
            # "action_chunks": np.stack(action_list)
        }
        
        # RawSaver needs observation N+1. If the loop ended normally, we might have it.
        # If stopped early, we might need to append the last obs again or handle it.
        # For simplicity, let's just save the N observations collected during the N steps.
        # Check RawSaver documentation if N+1 is strictly required.
        # If N+1 is needed, you might need to get one final observation after the loop.

        saver.save_traj(episode_idx, agent_data=agent_data, obs_dict=obs_dict)
        print("Trajectory saved successfully.")
    except Exception as e:
        print(f"Error saving trajectory {episode_idx}: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Run inference on USC WidowX robot using OpenPI policy server.")
    parser.add_argument(
        "--policy-server-address",
        type=str,
        default="https://whippet-pet-singularly.ngrok.app",
        help="Address (host:port) of the policy server.",
    )
    parser.add_argument("--robot-ip", type=str, default="localhost", help="IP address of the WidowX robot controller.")
    parser.add_argument("--robot-port", type=int, default=5556, help="IP address of the WidowX robot controller.")
    parser.add_argument("--cameras", nargs='+', default=["external", "over_shoulder"], help="List of camera names to use (e.g., external over_shoulder). Should match policy expectations.")
    parser.add_argument("--prompt", type=str, required=True, help="Task prompt for the policy.")
    parser.add_argument(
        "--max-action-length",
        type=int,
        default=10,
        help="Maximum number of actions to execute regardless of action chunk size from policy. Can use to query policy more often.",
    )
    parser.add_argument("--hz", type=int, default=5, help="Control frequency.")
    parser.add_argument("--save-dir", type=str, default="./trajectory_data/usc_widowx", help="Directory to save trajectory data.")
    args = parser.parse_args()

    # --- Initialization ---
    policy_client = None
    widowx_client = None
    print(f"Attempting to connect to policy server at {args.policy_server_address}...")

    # Re-introduce URL parsing
    parsed_url = urlparse(args.policy_server_address)
    host = parsed_url.hostname
    port = parsed_url.port

    # Initialize with parsed host and port
    policy_client = _websocket_client_policy.WebsocketClientPolicy(address=args.policy_server_address)

    # Optional: Add a ping or status check here if the client supports it
    print(f"Policy client initialized for host '{host}' on port {port}.")

    widowx_client = init_robot(args.robot_ip, args.robot_port)

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    saver = RawSaver(str(save_path))
    print(f"Saving trajectories to: {save_path.resolve()}")

    episode_idx = 0
    while True:
        print(f"\n--- Starting Episode {episode_idx} ---")
        print(f"Prompt: {args.prompt}")

        input("Press Enter to start the episode...")

        reset_requested = run_inference_loop(args, policy_client, widowx_client, saver, episode_idx)

        if not reset_requested:
            # If stop was requested or loop finished normally, stop the script
            print("Exiting inference script.")
            break
        else:
            # If reset was requested, increment episode index and continue
            episode_idx += 1
            print("\nResetting for next episode...")
            time.sleep(1.0)  # Pause before starting next


if __name__ == "__main__":
    main() 