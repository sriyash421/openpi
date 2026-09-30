"""Chained LIBERO rollouts with inline VLM feedback (runs in libero.sif, py3.8, next to a policy server).

For each assigned task: `attempts` attempts from one random init state; attempt j sees attempt j-1's 8 frames (tiled
into the context image) and its judge feedback. Writes per-attempt records.jsonl + frames npz + video mp4.

context_mode: real (default) | real_noinstr (real, but retries drop the instruction: feedback is the only goal text) |
empty (never pass context: the no-feedback baseline) | shuffled (pass the previous attempt of a *different* task, a
counterfactual that tests whether the policy reads the context) | generic (real frames, feedback replaced by a
correction-free success/failure line) | textonly (blank frames, real feedback) | null (blank frames + generic text).
textonly/null still set has_context, so they isolate frames vs text with the context path switched on.
"""
import collections
import dataclasses
import json
import logging
import pathlib
import random
import sys
import time

import imageio
import numpy as np
import tyro
from libero.libero import benchmark
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

sys.path.insert(0, "/app/examples/libero")
from main import LIBERO_DUMMY_ACTION, LIBERO_ENV_RESOLUTION, _get_libero_env, _quat2axisangle  # noqa: E402

sys.path.insert(0, "/app")
from onpolicy import context  # noqa: E402
from onpolicy.judge import SUCCESS_FEEDBACK, Judge  # noqa: E402

GENERIC_FAILURE = "The previous attempt failed."

MAX_STEPS = {"libero_90": 400, "libero_10": 520}


@dataclasses.dataclass
class Args:
    port: int = 8000
    task_suite_name: str = "libero_90"
    tasks: str = "0"  # comma-separated task ids handled by this client
    attempts: int = 4
    context_mode: str = "real"
    judge_model: str = "google/gemini-3.8-flash"
    out_dir: str = "rollouts"
    seed: int = 0
    replan_steps: int = 5
    num_steps_wait: int = 10
    negatives: int = 16  # (s_i, a_i) replan points saved per failed attempt, for failure suppression in training


def run_attempt(env, init_state, client, task_description, ctx_image, prompt, max_steps, args):
    env.reset()
    obs = env.set_init_state(init_state)
    plan, frames, views, done, t, queries = collections.deque(), [], [], False, 0, []
    while t < max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
        frames.append(img)
        views.append(np.concatenate([img, wrist], axis=1))
        if not plan:
            element = {
                "observation/image": img,
                "observation/wrist_image": wrist,
                "observation/state": np.concatenate((obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])),
                "observation/context_image": ctx_image if ctx_image is not None else context.empty_context(),
                "observation/has_context": np.bool_(ctx_image is not None),
                "prompt": prompt,
            }
            chunk = np.asarray(client.infer(element)["actions"])
            queries.append((img, wrist, element["observation/state"], chunk))
            plan.extend(chunk[: args.replan_steps])
        obs, _, done, _ = env.step(plan.popleft().tolist())
        if done:
            break
        t += 1
    # The loop records frames before each step; add the final state (on success, the frame that shows it).
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
    frames.append(img)
    views.append(np.concatenate([img, wrist], axis=1))
    return bool(done), t, frames, views, queries


def main(args: Args) -> None:
    out = pathlib.Path(args.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    for retry in range(20):  # ~30 clients per server connect at once; a handshake can time out under that load
        try:
            client = _websocket_client_policy.WebsocketClientPolicy("127.0.0.1", args.port)
            break
        except TimeoutError:
            time.sleep(3 + retry)
    judge = Judge(model=args.judge_model)
    rng = random.Random(args.seed)
    records = open(out / f"records_{args.tasks.replace(',', '-')}.jsonl", "a")
    donor = None  # last attempt of the previous task, for context_mode=shuffled

    for ci, task_id in enumerate(int(x) for x in args.tasks.split(",")):
        chain = f"{args.seed}-{ci}"  # unique per chain: a client may run several chains of the same task (evals)
        task = suite.get_task(task_id)
        env, description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        inits = suite.get_task_init_states(task_id)
        init_idx = rng.randrange(len(inits))
        prev = None  # (8 frames, feedback) of the previous attempt of this task
        for k in range(args.attempts):
            src = donor if args.context_mode == "shuffled" else prev
            use = src if (args.context_mode != "empty" and src is not None) else None
            if use and args.context_mode in ("generic", "null"):
                use = (use[0], SUCCESS_FEEDBACK if use[2] else GENERIC_FAILURE, use[2])
            blank = args.context_mode in ("textonly", "null")
            ctx_img = (context.empty_context() if blank else context.tile(use[0])) if use else None
            # real_noinstr: retries get the feedback without the instruction, matching training-time instruction dropout
            instr = None if (use and args.context_mode == "real_noinstr") else description
            prompt = context.prompt(instr, use[1] if use else None)
            success, steps, frames, views, queries = run_attempt(env, inits[init_idx], client, description, ctx_img, prompt, MAX_STEPS[args.task_suite_name], args)
            sub = context.subsample(frames)
            if args.context_mode in ("generic", "null"):  # feedback text is never shown: skip the VLM
                verdict = {"success": success, "feedback": SUCCESS_FEEDBACK if success else GENERIC_FAILURE, "cost": 0.0, "error": None}
            else:
                verdict = judge(description, sub, sim_success=success)  # VLM only explains failures
            name = f"task{task_id:02d}_c{chain}_a{k}_{'success' if success else 'failure'}"
            np.savez_compressed(out / "frames" / f"{name}.npz", frames=np.stack(sub))
            neg = None
            if not success and args.negatives > 0:
                picks = sorted(rng.sample(range(len(queries)), min(args.negatives, len(queries))))
                neg = str(out / "frames" / f"{name}_neg.npz")
                np.savez_compressed(neg, **{key: np.stack([queries[j][c] for j in picks]).astype(dt)
                                            for c, (key, dt) in enumerate([("image", np.uint8), ("wrist", np.uint8),
                                                                           ("state", np.float32), ("actions", np.float32)])})
            imageio.mimwrite(out / "videos" / f"{name}.mp4", views, fps=10)
            records.write(json.dumps({"suite": args.task_suite_name, "task_id": task_id, "task": description, "chain": chain, "attempt": k,
                                      "init": init_idx, "context_mode": args.context_mode, "had_context": use is not None,
                                      "success": success, "steps": steps, "judge_success": verdict["success"],
                                      "feedback": verdict["feedback"], "judge_cost": verdict["cost"], "judge_error": verdict["error"],
                                      "frames": str(out / "frames" / f"{name}.npz"), "video": str(out / "videos" / f"{name}.mp4"), "negatives": neg}) + "\n")
            records.flush()
            logging.info(f"task {task_id} attempt {k}: success={success} judge={verdict['success']} fb={verdict['feedback']!r}")
            prev = (sub, verdict["feedback"], success)
        donor = prev
        env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
