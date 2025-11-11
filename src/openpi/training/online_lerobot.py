from __future__ import annotations

import json
import logging
import copy
import socketserver
import threading
from pathlib import Path
from typing import Callable, Optional
import pickle
import zlib
from typing import Any
import numpy as np
import time

import datasets
from datasets import concatenate_datasets
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    embed_images,
    hf_transform_to_torch,
)
from lerobot.common.datasets.compute_stats import get_feature_stats, sample_indices

# We extend LeRobotDataset and add a background socket listener that accepts per-episode
# binary payloads (length-prefixed, zlib-compressed pickle) and updates the underlying
# hf_dataset and metadata in-place.

# def compute_episode_stats(episode_data: dict[str, list[str] | np.ndarray], features: dict) -> dict:
#     ep_stats = {}
#     for key, data in episode_data.items():
#         if features[key]["dtype"] == "string":
#             continue  # HACK: we should receive np.arrays of strings
#         elif features[key]["dtype"] in ["image", "video"]:
#             indices = sample_indices(len(data))
#             print(features[key].items())
#             ep_ft_array = features[key][indices]
#             axes_to_reduce = (0, 2, 3)  # keep channel dim
#             keepdims = True
#         else:
#             ep_ft_array = data  # data is already a np.ndarray
#             axes_to_reduce = 0  # compute stats over the first axis
#             keepdims = data.ndim == 1  # keep as np.array

#         ep_stats[key] = get_feature_stats(ep_ft_array, axis=axes_to_reduce, keepdims=keepdims)

#         # finally, we normalize and remove batch dim for images
#         if features[key]["dtype"] in ["image", "video"]:
#             ep_stats[key] = {
#                 k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in ep_stats[key].items()
#             }

#     return ep_stats


class OnlineLeRobotDataset(LeRobotDataset):

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        remap_keys: dict[str, str] | None = None,
        drop_keys: list[str] | None = None,
        online_dataset_path: str | None = None,
        server_host: str = "localhost",
        server_port: int = 50051,
    ):
        """
        Extending LeRobotDataset to support online data recording via a background socket server.
            online_dataset_path (str | None, optional): If specified, this will start a background socket
                server that listens for incoming episode data to be added to the dataset. The dataset will
                be updated in-place as new episodes are received. Defaults to None.
            server_host (str, optional): Host address for the socket server. Defaults to "localhost".
            server_port (int, optional): Port number for the socket server. Defaults to 50051.
        """
        # Initialize base dataset normally; we assume existing timestamps are valid.
        super().__init__(
            repo_id,
            root,
            episodes,
            image_transforms,
            delta_timestamps,
            tolerance_s,
            revision,
            force_cache_sync,
            download_videos,
            video_backend,
            remap_keys,
            drop_keys,
        )
        self.online_dataset_path = Path(online_dataset_path) if online_dataset_path else None
        if self.online_dataset_path is not None:
            self.online_dataset_path.mkdir(parents=True, exist_ok=True)
        # Maintain separate write root without mutating the base dataset's root; we'll use online_dataset_path explicitly when saving.
        # self._write_root = self.online_dataset_path if self.online_dataset_path is not None else self.root
        self.server_host = server_host
        self.server_port = server_port
        # If an online dataset path is provided, direct future writes there by updating root.
        # if self.online_dataset_path is not None:
        #     self.root = self.online_dataset_path
        self.episode_buffer = self.create_episode_buffer()
        # Track last timestamp within the CURRENT episode to enforce monotonically increasing timestamps.
        self._last_timestamp = None

        ## print episode_data_index
        print(f"[online_ds] Initialized OnlineLeRobotDataset with {self.meta.total_episodes} episodes and {self.meta.total_frames} frames.")
        print(f"[online_ds] Episode data index length: {len(self.episode_data_index["from"])}")
    
    def start_socket_server(self) -> None:
        """Start a background socket server to listen for incoming episode data."""
        if self.online_dataset_path is None:
            raise ValueError("online_dataset_path must be specified to start the socket server.")

        class EpisodeRequestHandler(socketserver.BaseRequestHandler):
            def handle(handler_self):
                # Read the length prefix (4 bytes)
                length_bytes = handler_self.request.recv(4)
                if len(length_bytes) < 4:
                    logging.error("Received incomplete length prefix.")
                    return
                length = int.from_bytes(length_bytes, byteorder='big')

                # Read the compressed payload
                compressed_payload = b''
                while len(compressed_payload) < length:
                    chunk = handler_self.request.recv(length - len(compressed_payload))
                    if not chunk:
                        break
                    compressed_payload += chunk

                if len(compressed_payload) != length:
                    logging.error("Received incomplete payload.")
                    return

                # Decompress and unpickle the episode data
                # try:
                payload = zlib.decompress(compressed_payload)
                episode_data = pickle.loads(payload)
                done = episode_data.get('done', False)
                # Sanitize frame features: remove meta/system columns not expected by add_frame.
                frame_dict = dict(episode_data.get('frame', {}))
                for k in ('timestamp', 'index', 'frame_index', 'task_index', 'episode_index'):
                    frame_dict.pop(k, None)
                # Update the dataset with the new episode data
                # if "size" not in self.episode_buffer.keys():
                #     self.episode_buffer["size"] = len(self.episode_buffer['actions'])
                # Enforce monotonic timestamp progression per episode to avoid sync validation errors.
                incoming_ts = episode_data.get('timestamp')
                if self._last_timestamp is not None and incoming_ts is not None and incoming_ts <= self._last_timestamp:
                    incoming_ts = self._last_timestamp + (1.0 / max(self.fps, 1.0))
                self.add_frame(frame_dict, episode_data.get('task'), incoming_ts)
                self._last_timestamp = incoming_ts
                try:
                    cur_size = int(self.episode_buffer.get("size", -1))
                except Exception:
                    cur_size = -1
                logging.debug("[online_ds] added frame: size=%s ts=%s", cur_size, f"{self._last_timestamp:.6f}" if self._last_timestamp is not None else None)
                    # logging.info("Successfully added a frame to current episode buffer.")
                # except Exception as e:
                #     logging.error(f"Failed to process episode data: {e}")
                
                # try:
                if done:
                    episode_buffer = copy.deepcopy(self.episode_buffer)
                    # reset buffer immediately to start next episode cleanly
                    self.episode_buffer = self.create_episode_buffer()
                    length = episode_buffer.get("size", 0)
                    total_frames = self.meta.total_frames
                    episode_index = self.meta.total_episodes
                    logging.info(f"Saving episode {episode_index} of length {length} starting at frame {total_frames}.")
                    # Dump diagnostics before saving to catch any shape/length mismatches early
                    try:
                        episode_buffer.pop("task_index", None)  # remove task_index to avoid confusion
                        episode_buffer.pop("index", None)  # remove task_index to avoid confusion
                        self._dump_episode_buffer_debug(episode_buffer)
                    except Exception as e:
                        logging.warning("[online_ds] Failed to dump episode buffer debug: %s", e)
                    if length <= 0:
                        logging.warning("[online_ds] done=True received but episode size is 0; skipping save.")
                    else:
                        # Save using overridden logic writing to self._write_root
                        self.save_episode(episode_buffer)
                        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
                        print(f"[online_ds] Episode {episode_index} saved successfully.")
                        print(f"[online_ds] Dataset now has {self.meta.total_episodes} episodes and {self.meta.total_frames} frames.")
                        print(f"[online_ds] Episode data index updated to {len(self.episode_data_index["from"])}.")
                        time.sleep(1.0)
                    # self.save_episode(episode_buffer)
                    # logging.info("Episode saved (done=True received).")
                    # Reset tracking for next episode
                    self._last_timestamp = None
                    # self.episode_buffer = self.create_episode_buffer()    
                # except Exception as e:
                #     logging.error(f"Failed to save episode: {e}")

        server = socketserver.ThreadingTCPServer((self.server_host, self.server_port), EpisodeRequestHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        logging.info(f"Socket server started on {self.server_host}:{self.server_port}.")
    
    def save_episode(self, episode_buffer: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        # if not episode_data:
        #     episode_buffer = self.episode_buffer
        # else:
            # episode_buffer = episode_data

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])
        # validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)
        # episode_buffer["task_index"] = np.full((episode_length,), episode_buffer["task_index"][0])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        # ep_stats = compute_episode_stats(episode_buffer, self.features)
        # hack to keep stats the same
        # ep_stats = copy.deepcopy(self.meta.stats)
        # self.meta.stats = None

        # if len(self.meta.video_keys) > 0:
        #     video_paths = self.encode_episode_videos(episode_index)
        #     for key in self.meta.video_keys:
        #         episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos

        # TODO: figure out why data saving is failing without this change? or just debug it once.
        self.meta.save_episode(episode_index, episode_length, episode_tasks, {})

        # ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        # ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}

        #TODO: fix sync later
        # check_timestamps_sync(
        #     episode_buffer["timestamp"],
        #     episode_buffer["episode_index"],
        #     ep_data_index_np,
        #     self.fps,
        #     self.tolerance_s,
        # )

        # video_files = list(self.root.rglob("*.mp4"))
        # assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        # parquet_files = list(self.root.rglob("*.parquet"))
        # assert len(parquet_files) == self.num_episodes

        # # delete images
        # img_dir = self.root / "images"
        # if img_dir.is_dir():
        #     shutil.rmtree(self.root / "images")

        # if not episode_data:  # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def _dump_episode_buffer_debug(self, episode_buffer: dict) -> None:
        """Log keys with list lengths and array shapes prior to saving an episode."""
        exp_len = episode_buffer.get("size", None)
        lines = [
            "[online_ds] Episode buffer diagnostics:",
            f"  expected size: {exp_len}",
        ]
        mismatches = []
        for k in sorted(episode_buffer.keys()):
            v = episode_buffer[k]
            if isinstance(v, list):
                vlen = len(v)
                shape = getattr(v[0], "shape", None) if v and hasattr(v[0], "shape") else None
            elif isinstance(v, np.ndarray):
                vlen = v.shape[0] if v.ndim > 0 else 1
                shape = v.shape
            else:
                vlen = "scalar"
                shape = None
            lines.append(f"  - {k}: len={vlen} shape={shape}")
            if isinstance(vlen, int) and exp_len is not None and vlen != exp_len and k not in ("episode_index", "index"):
                mismatches.append((k, vlen))
        if mismatches:
            lines.append("  MISMATCHES:")
            for k, vlen in mismatches:
                lines.append(f"    * {k}: len={vlen} != expected {exp_len}")
        logging.info("\n".join(lines))