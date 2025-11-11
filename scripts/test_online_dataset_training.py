#!/usr/bin/env python3
"""
Test harness for OnlineLeRobotDataset.

This script does two things:
- Starts a background process that reads episodes from --source-online-dir and
    sends episode dicts over a socket to the OnlineLeRobotDataset listener,
    one episode at a time with a delay.
- Builds a TrainConfig programmatically, injects online_dataset_dir into the
    data config (this is where metadata will be written), and runs training via
    scripts.train.main(...).

Usage (example):
  python scripts/test_online_dataset_training.py \
    --config-name pi0_libero_low_mem_finetune_sriyash \
    --base-repo /gscratch/socialrl/sriyash/openpi/data/task_57_libero \
    --source-online-dir /gscratch/socialrl/sriyash/openpi/data/task_57_libero \
    --online-dir /tmp/openpi-online-libero \
    --exp-name online-test \
    --num-train-steps 200

Notes:
- You should point --source-online-dir to a valid LeRobot dataset. The streamer
    will stream episode dicts over TCP to the online dataset.
- This script disables wandb and writes checkpoints under ./checkpoints/<name>/<exp> by default.
- If your chosen config requires normalization stats, ensure the corresponding
  assets exist. Otherwise, you may adapt the config or compute stats first.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import sys
import time
from pathlib import Path
import json
import socket
import pickle
import zlib
from datetime import datetime

# Ensure repo root is on sys.path when running directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi.training import config as _config
from scripts import train as _train


def _stream_send(src_root: Path, online_root: Path | None, delay_s: float = 0.05, host: str = "127.0.0.1", port: int = 9999):
    """Stream frames from existing episodes as per-frame payloads.

    Each frame payload schema:
        {
          'frame': {<column_key>: value, 'episode_index': int},
          'task': <task_string>,
          'timestamp': float (seconds),
          'done': bool (True iff last frame of episode)
        }
    """
    src_meta = src_root / "meta"
    info_path = src_meta / "info.json"
    episodes_path = src_meta / "episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise FileNotFoundError(f"Missing meta files under {src_meta}")
    with info_path.open("r") as f:
        info = json.load(f)
    data_path_tmpl = info.get("data_path")
    chunk_size = int(info.get("chunks_size", 1000))
    fps = info.get("fps", 30)
    if not data_path_tmpl:
        raise ValueError("info.json missing data_path template")

    def _connect_retry() -> socket.socket:
        start = time.time()
        while True:
            try:
                return socket.create_connection((host, port), timeout=2.0)
            except OSError:
                if time.time() - start > 30.0:
                    raise
                time.sleep(0.25)

    # Determine expected feature keys from the source dataset (exclude meta-managed index fields)
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset as _LRD
        src_ds = _LRD(repo_id=str(src_root), root=str(src_root))
        expected_keys = [k for k in src_ds.features.keys() if k not in ("index", "episode_index", "task_index")]
        print(f"[streamer] Expected feature keys: {expected_keys}")
    except Exception as e:
        print(f"[streamer] Could not read source dataset features, will send all columns; error: {e}")
        expected_keys = None

    # Load episodes list
    with episodes_path.open("r") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep_index = int(ep.get("episode_index", -1))
            if ep_index < 0:
                continue
            ep_len = int(ep.get("length", 0))
            tasks = ep.get("tasks", []) or ["unknown"]
            task = tasks[0]
            ep_chunk = ep_index // chunk_size
            rel_data_path = data_path_tmpl.format(episode_chunk=ep_chunk, episode_index=ep_index)
            parquet_file = (src_root / rel_data_path).resolve()
            try:
                from datasets import load_dataset as hf_load_dataset
                ds = hf_load_dataset("parquet", data_files=[str(parquet_file)], split="train")
                if "episode_index" in ds.column_names:
                    ds = ds.filter(lambda row: row["episode_index"] == ep_index)
                ds_np = ds.with_format("numpy")
            except Exception as e:
                print(f"[streamer] Failed loading parquet {parquet_file}: {e}")
                continue

            # Iterate frame by frame
            for i in range(min(ep_len, len(ds_np))):
                # Indexing returns a dict of column -> value for this row with numpy types as requested
                full_row = dict(ds_np[i])
                # Build frame strictly from expected_keys if available, else send all columns
                if expected_keys is not None:
                    frame = {k: full_row[k] for k in expected_keys if k in full_row}
                    # Debug missing/extra keys
                    missing = set(expected_keys) - set(frame.keys())
                    extra = set(full_row.keys()) - set(expected_keys)
                    if missing:
                        print(f"[streamer] ep={ep_index} frame={i}: missing={sorted(missing)}")
                    if extra:
                        # benign, the server strips meta keys anyway
                        pass
                else:
                    frame = full_row
                # Always annotate episode index for context (server strips meta appropriately)
                frame['episode_index'] = ep_index
                timestamp = i / fps
                payload = {
                    'frame': frame,
                    'task': task,
                    'timestamp': float(timestamp),
                    'done': i == ep_len - 1,
                }
                blob = zlib.compress(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
                try:
                    with _connect_retry() as sock:
                        sock.sendall(len(blob).to_bytes(4, 'big') + blob)
                        # no ack expected
                except Exception as e:
                    print(f"[streamer] send failed frame {i} ep {ep_index}: {e}")
                    break
                time.sleep(delay_s)
            time.sleep(2.0)  # delay between episodes
    print("[streamer] Completed streaming all frames.")


def _make_train_config(
    cfg_name: str,
    online_dir: str | None,
    exp_name: str,
    num_train_steps: int,
    server_host: str,
    server_port: int,
) -> _config.TrainConfig:
    # Look up the base train config by name.
    base_cfg = None
    for c in getattr(_config, "_CONFIGS"):
        if c.name == cfg_name:
            base_cfg = c
            break
    if base_cfg is None:
        raise ValueError(f"Config named {cfg_name!r} not found in openpi.training.config._CONFIGS")

    # Inject online_dataset_dir while preserving any existing base_config flags (e.g., prompt_from_task).
    orig_base = getattr(base_cfg.data, "base_config", None) or _config.DataConfig()
    new_base = dataclasses.replace(
        orig_base,
        online_dataset_dir=online_dir,
        server_host=server_host,
        server_port=server_port,
    )
    # data_factory = dataclasses.replace(base_cfg.data, repo_id=base_repo, base_config=new_base)
    data_factory = dataclasses.replace(base_cfg.data, base_config=new_base)

    # Create a lightweight training config for quick testing.
    train_cfg = dataclasses.replace(
        base_cfg,
        exp_name=exp_name,
        data=data_factory,
        validation_data=None,
        wandb_enabled=False,
        num_train_steps=num_train_steps,
        log_interval=min(10, max(1, num_train_steps // 10)),
        save_interval=max(1000, num_train_steps + 1),  # don't save mid-run on tiny tests
        batch_size=min(16, base_cfg.batch_size),
        num_workers=min(4, base_cfg.num_workers),
        overwrite=True,
    )
    return train_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, help="Name of TrainConfig in _CONFIGS to run.")
    parser.add_argument("--source-online-dir", required=True, help="Existing LeRobot dataset to stream from.")
    parser.add_argument(
        "--online-dir",
        required=False,
        default=None,
        help="Optional directory to persist received payloads. If omitted, data is kept in-memory only.",
    )
    parser.add_argument("--exp-name", default="online-test", help="Experiment name for checkpoints.")
    parser.add_argument("--num-train-steps", type=int, default=200)
    parser.add_argument("--copy-delay", type=float, default=0.5, help="Delay in seconds between sending episodes.")
    parser.add_argument("--online-host", type=str, default="127.0.0.1", help="Host for online dataset listener.")
    parser.add_argument("--online-port", type=int, default=9999, help="Port for online dataset listener.")
    args = parser.parse_args()

    src = Path(args.source_online_dir).resolve()
    online_dir = args.online_dir
    dst = Path(online_dir).resolve() if online_dir is not None else None
    if not src.exists():
        raise FileNotFoundError(f"source-online-dir not found: {src}")

    # Clean/prepare destination dir (meta will be written here by the online dataset).
    if online_dir is not None:
        # Only clean if the user explicitly provided the directory
        if dst.exists():  # type: ignore[union-attr]
            print(f"[runner] Cleaning existing online dir: {dst}")
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    # If no directory is provided, do not create or write anything.

    # Build config
    cfg = _make_train_config(
        cfg_name=args.config_name,
        online_dir=str(dst) if dst is not None else None,
        exp_name=args.exp_name,
        num_train_steps=args.num_train_steps,
        server_host=args.online_host,
        server_port=args.online_port,
    )

    # Start background sender to stream episode dicts to the listener.
    import multiprocessing as mp

    streamer = mp.Process(
        target=_stream_send,
    args=(src, dst, args.copy_delay, args.online_host, args.online_port),
        daemon=True,
    )
    streamer.start()
    print(f"[runner] Streaming episodes from {src} -> socket {args.online_host}:{args.online_port} (pid={streamer.pid})")

    # Start training (listener will be up shortly; sender retries connect)
    print("[runner] Starting training with online dataset...")
    _train.main(cfg)

    print("[runner] Training finished. Waiting for streamer to exit...")
    streamer.join(timeout=5.0)
    if streamer.is_alive():
        print("[runner] Streamer still alive; terminating.")
        streamer.terminate()


if __name__ == "__main__":
    main()

# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/test_online_dataset_training.py --config-name pi0_libero_low_mem_finetune_sriyash --source-online-dir data/task_57_libero --online-dir data/stream_task_57 --copy-delay 30