#!/usr/bin/env python3
"""Minimal standalone script to:
  1. Instantiate an OnlineLeRobotDataset backed by an existing *base* LeRobot dataset directory.
  2. Start the socket listener (length‑prefixed, zlib+pickle protocol matching `OnlineLeRobotDataset`).
  3. Stream EXACTLY TWO episodes frame‑by‑frame to the listener.
  4. Print rich debug info after each send and after the listener saves an episode.

USAGE (example):
  python scripts/online_stream_two_episodes.py \
      --base-repo /gscratch/socialrl/sriyash/openpi/data/task_44 \
      --online-dir /tmp/online_test_task_44 \
      --host 127.0.0.1 --port 50111 \
      --delay 0.02

NOTES:
- --base-repo must be a *local* LeRobot dataset directory containing meta/info.json & meta/episodes.jsonl.
- The script copies NO data initially; it just uses base-repo to discover features & fps.
- We purposely only stream scalar/tensor features (no raw image/video blobs). If your dataset contains image
  features (dtype 'image'/'video') you will need to adapt sending (currently ignored here) or rely on your
  dataset's add_frame implementation to handle image paths if required.
- Keys automatically stripped before sending: timestamp/index/frame_index/task_index/episode_index/task.
- If you see FEATURE MISMATCH errors in the server logs, this script will print a diff to help you adjust.

Debug Strategy:
- For each frame we print: frame i / len, episode idx, sent_keys, missing_keys (expected - sent), extra_keys.
- After each frame we poll (with a short sleep) the dataset meta to show growing total_frames/episodes.
- After final frame (done=True) we wait a short time then show updated meta & list parquet files created.

"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import socket
import time
import zlib
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
import numpy as np

from openpi.training.online_lerobot import OnlineLeRobotDataset

META_STRIP = {"timestamp", "index", "frame_index", "task_index", "episode_index", "task"}


def _length_prefix_send(host: str, port: int, payload: dict):
    blob = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    with socket.create_connection((host, port), timeout=5) as s:
        s.sendall(len(blob).to_bytes(4, "big") + blob)


def _choose_two_episodes(meta_dir: Path) -> list[dict]:
    episodes_path = meta_dir / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes file: {episodes_path}")
    selected = []
    with episodes_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            selected.append(ep)
            if len(selected) == 2:
                break
    if len(selected) < 2:
        raise RuntimeError("Base dataset has fewer than 2 episodes.")
    return selected


def _load_episode_frames(base_root: Path, info: dict, ep_index: int) -> tuple[dict[str, np.ndarray], int]:
    chunk_size = int(info.get("chunks_size", 1000))
    tmpl = info.get("data_path")
    if not tmpl:
        raise ValueError("info.json missing data_path")
    ep_chunk = ep_index // chunk_size
    rel_path = tmpl.format(episode_chunk=ep_chunk, episode_index=ep_index)
    parquet_file = base_root / rel_path
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet for episode {ep_index} not found: {parquet_file}")
    ds = load_dataset("parquet", data_files=[str(parquet_file)], split="train")
    if "episode_index" in ds.column_names:
        ds = ds.filter(lambda row: row["episode_index"] == ep_index)
    ds_np = ds.with_format("numpy")
    return {col: ds_np[col] for col in ds_np.column_names}, len(ds_np)


def stream_two(args):
    base_root = Path(args.base_repo).resolve()
    # Use a distinct directory for the ONLINE dataset so we don't corrupt / modify the base dataset.
    online_root = Path(args.online_dir).resolve()
    if not base_root.exists():
        raise FileNotFoundError(f"Base repo path does not exist: {base_root}")
    (online_root).mkdir(parents=True, exist_ok=True)

    meta_dir = base_root / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Missing info.json in {meta_dir}")
    info = json.loads(info_path.read_text())
    fps = float(info.get("fps", 30.0))

    print(f"[init] Base repo: {base_root}\n       Online root: {online_root}\n       FPS: {fps}")

    ds = OnlineLeRobotDataset(
        repo_id=str(base_root),
        root=str(base_root),  # initialize features/meta from existing dataset
        online_dataset_path=str(online_root),  # write new data separately
        server_host=args.host,
        server_port=args.port,
    )
    # Start server
    ds.start_socket_server()
    print(f"[listener] Started on {args.host}:{args.port}")

    # Inspect expected feature keys (include image/video types; skip meta-managed columns)
    expected_keys = []
    for k, ft in ds.features.items():
        if k in ("index", "episode_index", "task_index", "timestamp"):
            continue
        expected_keys.append(k)
    print(f"[features] Expecting frame keys: {expected_keys}")

    episodes = _choose_two_episodes(meta_dir)
    print("[episodes] Selected indices:", [e["episode_index"] for e in episodes])

    for ep_pos, ep in enumerate(episodes):
        ep_index = int(ep["episode_index"])
        ep_tasks = ep.get("tasks", []) or ["unknown task"]
        task = ep_tasks[0]
        frame_arrays, ep_len = _load_episode_frames(base_root, info, ep_index)
        print(f"[episode {ep_pos}] index={ep_index} len={ep_len} task='{task}'")

        for i in range(ep_len):
            # Build frame dict from expected keys only
            frame = {}
            for k in expected_keys:
                if k not in frame_arrays:
                    continue
                arr = frame_arrays[k]
                # Strict per-frame extraction; no fallback to full array
                try:
                    val = arr[i]
                except Exception:
                    continue  # skip non-indexable columns
                # Convert 0-dim numpy scalars to Python types
                if isinstance(val, np.ndarray) and val.shape == ():
                    try:
                        val = val.item()
                    except Exception:
                        pass
                frame[k] = val
            # Compute diffs for debugging
            sent_keys = set(frame.keys())
            missing = set(expected_keys) - sent_keys
            extra = sent_keys - set(expected_keys)
            if missing or extra:
                print(f"  [warn frame {i}] missing={missing} extra={extra}")

            payload = {
                "frame": frame,  # server sanitizes meta keys anyway
                "task": task,
                "timestamp": i / fps,
                "done": i == ep_len - 1,
            }
            try:
                _length_prefix_send(args.host, args.port, payload)
            except Exception as e:
                print(f"  [error] send frame {i} ep {ep_index}: {e}")
                break
            print(f"  [sent] ep={ep_index} frame={i+1}/{ep_len} keys={sorted(frame.keys())} done={payload['done']}")
            time.sleep(args.delay)

        # Allow server to flush & compute stats
        time.sleep(0.5)
        # After episode, print current meta (if updated)
        try:
            print(
                f"[meta] After episode {ep_index}: total_frames={ds.meta.total_frames} total_episodes={ds.meta.total_episodes}"
            )
        except Exception as e:
            print(f"[meta] Could not read meta yet: {e}")

    # Final inspection of written parquet files
    written_parquets = list(online_root.rglob("*.parquet"))
    print(f"[done] Parquet files written: {len(written_parquets)}")
    for p in written_parquets[:5]:
        print("   ", p.relative_to(online_root))
    if len(written_parquets) > 5:
        print("   ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-repo", required=True, help="Existing LeRobot dataset directory to source features from.")
    ap.add_argument("--online-dir", required=True, help="Directory to write the online dataset (will be created).")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Host interface for the OnlineLeRobotDataset socket server.")
    ap.add_argument("--port", type=int, default=50055, help="Port for the socket server.")
    ap.add_argument("--delay", type=float, default=0.02, help="Delay (s) between frames.")
    args = ap.parse_args()
    stream_two(args)


if __name__ == "__main__":
    main()
