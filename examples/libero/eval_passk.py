"""pass@k LIBERO eval: examples/libero/main.py with sharding, k rollouts per initial state, per-episode videos + JSON.

Rollout logic (preprocessing, wait steps, replanning, max steps, success) is unchanged from main.py.
Episodes = (task, init_state, rollout) for the first `num_inits` initial states of every task, `k` rollouts each.
This client runs episodes [shard::num_shards] of that list and writes one JSON line per episode.
"""
import collections
import dataclasses
import json
import logging
import pathlib

import imageio
from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

from main import LIBERO_DUMMY_ACTION
from main import LIBERO_ENV_RESOLUTION
from main import _get_libero_env
from main import _quat2axisangle

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    task_suite_name: str = "libero_10"
    num_steps_wait: int = 10
    num_inits: int = 5  # first N LIBERO initial states per task
    k: int = 8  # rollouts per initial state
    task_start: int = 0  # task range handled by this job: [task_start, task_end)
    task_end: int = -1
    shard: int = 0  # this client's slice of the job's episodes
    num_shards: int = 1
    out_dir: str = "data/libero/passk"
    seed: int = 7


def eval_passk(args: Args) -> None:
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_end = task_suite.n_tasks if args.task_end < 0 else args.task_end
    episodes = [(t, i, r) for t in range(args.task_start, task_end) for i in range(args.num_inits) for r in range(args.k)]
    episodes = episodes[args.shard :: args.num_shards]
    out = pathlib.Path(args.out_dir)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    results = open(out / f"results_{args.task_start}_{task_end}_shard{args.shard}.jsonl", "a")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    max_steps = MAX_STEPS[args.task_suite_name]

    env, cur_task = None, None
    for task_id, init_idx, rollout in episodes:
        task = task_suite.get_task(task_id)
        if task_id != cur_task:
            if env is not None:
                env.close()
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            initial_states = task_suite.get_task_init_states(task_id)
            cur_task = task_id

        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[init_idx])
        t, done, replay_images = 0, False, []
        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(np.concatenate([img, wrist_img], axis=1))
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                        "prompt": str(task_description),
                    }
                    action_plan.extend(client.infer(element)["actions"][: args.replan_steps])
                obs, reward, done, info = env.step(action_plan.popleft().tolist())
                if done:
                    break
                t += 1
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        name = f"task{task_id:02d}_init{init_idx}_r{rollout}_{'success' if done else 'failure'}.mp4"
        imageio.mimwrite(out / "videos" / name, replay_images, fps=10)
        results.write(json.dumps({"suite": args.task_suite_name, "task_id": task_id, "task": task_description,
                                  "init": init_idx, "rollout": rollout, "success": bool(done), "steps": t,
                                  "video": name}) + "\n")
        results.flush()
        logging.info(f"task {task_id} init {init_idx} r{rollout}: {'success' if done else 'failure'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_passk)
