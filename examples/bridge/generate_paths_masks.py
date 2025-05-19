"""
Script to generate paths and masks for Bridge data using VLM inference.
This script should be run before convert_bridge_data_to_lerobot.py to prepare the path/mask data.
"""

import dataclasses
import logging
import pathlib
from pathlib import Path
import tensorflow_datasets as tfds
import numpy as np
import tqdm
import tyro
import h5py
from src.openpi.policies.eval_maskpath_utils import get_path_mask_from_vlm


@dataclasses.dataclass
class Args:
    data_dir: str  # Directory containing the Bridge dataset
    output_dir: str  # Directory to save the generated paths and masks
    vlm_server_ip: str = "http://0.0.0.0:8000"  # VLM server address
    resize_size: int = 224  # Size to resize images for VLM
    draw_path: bool = True  # Whether to generate paths
    draw_mask: bool = True  # Whether to generate masks
    flip_image_horizontally: bool = False  # Whether to flip images horizontally


def generate_paths_masks(args: Args) -> None:
    """Generate paths and masks for Bridge data using VLM inference."""
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load Bridge dataset
    raw_dataset = tfds.load("bridge_v2", data_dir=args.data_dir, split="train")

    # Create HDF5 file to store paths and masks
    h5_path = output_path / "bridge_paths_masks.h5"
    with h5py.File(h5_path, "w") as f:
        # Process each episode
        for episode_idx, episode in enumerate(tqdm.tqdm(raw_dataset, desc="Processing episodes")):
            # Create group for this episode
            episode_group = f.create_group(f"episode_{episode_idx}")

            # Get task description
            task_description = episode["language_instruction"].decode()
            episode_group.attrs["task_description"] = task_description

            # Initialize arrays to store paths and masks
            num_steps = len(list(episode["steps"].as_numpy_iterator()))
            paths = []
            masks = []

            # Process each step
            for step in episode["steps"].as_numpy_iterator():
                # Get image from first available camera
                img = None
                for cam in ["image_0", "image_1", "image_2", "image_3"]:
                    if cam in step["observation"]:
                        img = step["observation"][cam]
                        break

                if img is None:
                    logging.warning(f"No valid image found in step {step['step_id']} of episode {episode_idx}")
                    continue

                # Preprocess image
                if args.flip_image_horizontally:
                    img = img[:, ::-1]

                # Get path and mask from VLM
                try:
                    # Get path and mask points without drawing them
                    _, path, mask = get_path_mask_from_vlm(
                        img,
                        "Center Crop",
                        task_description,
                        draw_path=False,  # Don't draw, just get points
                        draw_mask=False,  # Don't draw, just get points
                        verbose=False,
                        vlm_server_ip=args.vlm_server_ip,
                        path=None,  # Force new VLM query
                        mask=None,  # Force new VLM query
                    )

                    # Store path and mask points
                    if args.draw_path and path is not None:
                        paths.append(path)
                    if args.draw_mask and mask is not None:
                        masks.append(mask)

                except Exception as e:
                    logging.error(f"Error getting path/mask for step {step['step_id']} of episode {episode_idx}: {e}")
                    # Add empty path/mask to maintain alignment
                    if args.draw_path:
                        paths.append(np.array([]))
                    if args.draw_mask:
                        masks.append(np.array([]))

            # Save paths and masks for this episode
            if args.draw_path and paths:
                # Convert list of arrays to a single array with variable length
                max_path_len = max(len(p) for p in paths)
                padded_paths = np.zeros((len(paths), max_path_len, 2))
                for i, p in enumerate(paths):
                    if len(p) > 0:
                        padded_paths[i, : len(p)] = p
                episode_group.create_dataset("paths", data=padded_paths)
                episode_group.create_dataset("path_lengths", data=np.array([len(p) for p in paths]))

            if args.draw_mask and masks:
                # Convert list of arrays to a single array with variable length
                max_mask_len = max(len(m) for m in masks)
                padded_masks = np.zeros((len(masks), max_mask_len, 2))
                for i, m in enumerate(masks):
                    if len(m) > 0:
                        padded_masks[i, : len(m)] = m
                episode_group.create_dataset("masks", data=padded_masks)
                episode_group.create_dataset("mask_lengths", data=np.array([len(m) for m in masks]))

            # Save some metadata
            episode_group.attrs["num_steps"] = num_steps
            episode_group.attrs["has_paths"] = args.draw_path
            episode_group.attrs["has_masks"] = args.draw_mask

    logging.info(f"Generated paths and masks saved to {h5_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(generate_paths_masks)
