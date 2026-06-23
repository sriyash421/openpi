#!/usr/bin/env python3
"""Convert a chunked LeRobot v3-style YAM dataset to this repo's LeRobot v2.1 layout.

The LeRobot version pinned by this OpenPI checkout expects:
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/<video_key>/episode_000000.mp4
  meta/episodes.jsonl

The YAM dataset this script targets has a newer layout:
  data/chunk-000/file-000.parquet
  videos/<video_key>/chunk-000/file-000.mp4
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections.abc import Iterable
import json
from pathlib import Path
import re
import shutil
import subprocess

import av
import imageio_ffmpeg
import pandas as pd
from tqdm import tqdm


DEFAULT_SOURCE = Path("/gscratch/scrubbed/sriyash/yam_book_pick_shelf_place")
DEFAULT_DEST = Path("/gscratch/scrubbed/sriyash/yam_book_pick_shelf_place_lerobot")

VIDEO_DTYPE = "video"
V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--overwrite", action="store_true", help="Delete --dest first if it already exists.")
    parser.add_argument("--skip-videos", action="store_true", help="Only convert parquet and metadata.")
    parser.add_argument("--video-codec", default="libx264", help="ffmpeg encoder name for output episode videos.")
    parser.add_argument("--crf", default="23", help="CRF option for encoders that support it.")
    parser.add_argument("--ffmpeg", type=Path, default=None, help="Optional path to an ffmpeg executable.")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=4)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def copy_tree_contents(source: Path, dest: Path) -> None:
    if not source.exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def get_task(source: Path) -> str:
    tasks_path = source / "meta" / "tasks.jsonl"
    with tasks_path.open() as f:
        first = f.readline()
    return json.loads(first)["task"]


def get_video_keys(info: dict) -> list[str]:
    return [key for key, feature in info["features"].items() if feature["dtype"] == VIDEO_DTYPE]


def sorted_file_parts(directory: Path, suffix: str) -> list[Path]:
    def file_index(path: Path) -> int:
        match = re.search(r"file-(\d+)", path.stem)
        if match is None:
            raise ValueError(f"Expected a file-XXX name, got: {path}")
        return int(match.group(1))

    return sorted(directory.glob(f"file-*{suffix}"), key=file_index)


def load_source_table(source: Path) -> pd.DataFrame:
    parquet_files = sorted_file_parts(source / "data" / "chunk-000", ".parquet")
    if not parquet_files:
        raise FileNotFoundError(f"No source parquet files found under {source / 'data' / 'chunk-000'}")
    return pd.concat([pd.read_parquet(path) for path in parquet_files], ignore_index=True)


def write_episode_parquets(df: pd.DataFrame, dest: Path, chunks_size: int) -> dict[int, int]:
    episode_lengths = {}
    for episode_index, episode_df in tqdm(df.groupby("episode_index", sort=True), desc="Writing parquet"):
        episode_index = int(episode_index)
        episode_chunk = episode_index // chunks_size
        out_path = dest / V21_DATA_PATH.format(episode_chunk=episode_chunk, episode_index=episode_index)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        episode_df.to_parquet(out_path, index=False)
        episode_lengths[episode_index] = len(episode_df)
    return episode_lengths


def make_info(source_info: dict, episode_lengths: dict[int, int], *, include_videos: bool) -> dict:
    info = dict(source_info)
    info["codebase_version"] = "v2.0"
    info["total_episodes"] = len(episode_lengths)
    info["total_frames"] = int(sum(episode_lengths.values()))
    info["total_tasks"] = 1
    info["chunks_size"] = int(source_info.get("chunks_size", 1000))
    info["total_chunks"] = max(episode_lengths) // info["chunks_size"] + 1
    info["total_videos"] = len(get_video_keys(source_info)) * len(episode_lengths) if include_videos else 0
    info["splits"] = {"train": f"0:{len(episode_lengths)}"}
    info["data_path"] = V21_DATA_PATH
    info["video_path"] = V21_VIDEO_PATH if include_videos else None
    return info


def write_metadata(source: Path, dest: Path, source_info: dict, episode_lengths: dict[int, int], include_videos: bool) -> None:
    meta_dest = dest / "meta"
    meta_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "meta" / "stats.json", meta_dest / "stats.json")
    write_json(meta_dest / "info.json", make_info(source_info, episode_lengths, include_videos=include_videos))

    task = get_task(source)
    write_jsonl(meta_dest / "tasks.jsonl", [{"task_index": 0, "task": task}])
    write_jsonl(
        meta_dest / "episodes.jsonl",
        (
            {"episode_index": episode_index, "tasks": [task], "length": length}
            for episode_index, length in sorted(episode_lengths.items())
        ),
    )


class EpisodeVideoWriter:
    def __init__(
        self,
        dest: Path,
        video_key: str,
        episode_index: int,
        chunks_size: int,
        fps: int,
        codec: str,
        crf: str,
        ffmpeg: Path | None,
    ):
        episode_chunk = episode_index // chunks_size
        self._out_path = dest / V21_VIDEO_PATH.format(
            episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index
        )
        self._out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = fps
        self._codec = codec
        self._crf = crf
        self._ffmpeg = str(ffmpeg) if ffmpeg is not None else imageio_ffmpeg.get_ffmpeg_exe()
        self._process: subprocess.Popen | None = None
        self._closed = False

    def write(self, frame: av.VideoFrame) -> None:
        if self._process is None:
            self._process = subprocess.Popen(
                [
                    self._ffmpeg,
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{frame.width}x{frame.height}",
                    "-r",
                    str(self._fps),
                    "-i",
                    "-",
                    "-an",
                    "-vcodec",
                    self._codec,
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    self._crf,
                    "-preset",
                    "veryfast",
                    str(self._out_path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

        assert self._process.stdin is not None
        self._process.stdin.write(frame.to_ndarray(format="rgb24").tobytes())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is None:
            return

        assert self._process.stdin is not None
        self._process.stdin.close()
        stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed while writing {self._out_path}:\n{stderr}")


def open_episode_video(
    dest: Path,
    video_key: str,
    episode_index: int,
    chunks_size: int,
    fps: int,
    codec: str,
    crf: str,
    ffmpeg: Path | None,
) -> EpisodeVideoWriter:
    episode_chunk = episode_index // chunks_size
    out_path = dest / V21_VIDEO_PATH.format(
        episode_chunk=episode_chunk, video_key=video_key, episode_index=episode_index
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return EpisodeVideoWriter(dest, video_key, episode_index, chunks_size, fps, codec, crf, ffmpeg)


def split_camera_videos(
    source: Path,
    dest: Path,
    video_key: str,
    episode_ends: list[int],
    chunks_size: int,
    fps: int,
    codec: str,
    crf: str,
    ffmpeg: Path | None,
) -> None:
    source_videos = sorted_file_parts(source / "videos" / video_key / "chunk-000", ".mp4")
    if not source_videos:
        raise FileNotFoundError(f"No source videos found for {video_key}")

    episode_index = 0
    global_frame = 0
    writer = open_episode_video(dest, video_key, episode_index, chunks_size, fps, codec, crf, ffmpeg)
    try:
        for source_video in tqdm(source_videos, desc=f"Splitting {video_key}"):
            with av.open(str(source_video)) as input_container:
                for frame in input_container.decode(video=0):
                    next_episode_index = bisect_right(episode_ends, global_frame)
                    while episode_index < next_episode_index:
                        writer.close()
                        episode_index += 1
                        writer = open_episode_video(dest, video_key, episode_index, chunks_size, fps, codec, crf, ffmpeg)
                    writer.write(frame)
                    global_frame += 1
    finally:
        writer.close()

    if global_frame != episode_ends[-1]:
        raise ValueError(f"{video_key} had {global_frame} frames, expected {episode_ends[-1]}")


def split_videos(
    source: Path,
    dest: Path,
    info: dict,
    episode_lengths: dict[int, int],
    codec: str,
    crf: str,
    ffmpeg: Path | None,
) -> None:
    episode_ends = []
    total = 0
    for _, length in sorted(episode_lengths.items()):
        total += length
        episode_ends.append(total)

    for video_key in get_video_keys(info):
        split_camera_videos(
            source,
            dest,
            video_key,
            episode_ends,
            chunks_size=int(info.get("chunks_size", 1000)),
            fps=int(info["fps"]),
            codec=codec,
            crf=crf,
            ffmpeg=ffmpeg,
        )


def main() -> None:
    args = parse_args()
    if args.dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.dest} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(args.dest)

    args.dest.mkdir(parents=True)
    copy_tree_contents(args.source / "images", args.dest / "images")

    source_info = read_json(args.source / "meta" / "info.json")
    df = load_source_table(args.source)
    episode_lengths = write_episode_parquets(df, args.dest, chunks_size=int(source_info.get("chunks_size", 1000)))
    write_metadata(args.source, args.dest, source_info, episode_lengths, include_videos=not args.skip_videos)

    if not args.skip_videos:
        split_videos(args.source, args.dest, source_info, episode_lengths, args.video_codec, args.crf, args.ffmpeg)

    print(f"Converted dataset written to: {args.dest}")


if __name__ == "__main__":
    main()
