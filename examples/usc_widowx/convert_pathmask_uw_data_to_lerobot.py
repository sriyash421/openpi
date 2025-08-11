"""
Minimal script to create a LeRobot dataset with path/mask overlays for UW WidowX,
using an existing LeRobot dataset as input and an HDF5 file containing paths/masks.

This mirrors the simple path/mask overlay logic from
`examples/libero/convert_pathmask_libero_data_to_lerobot_vlm_preds.py`, but instead of
reading TFDS, it iterates frames from an existing LeRobot dataset and applies per-episode
paths/masks stored in an HDF5 file using the naming convention from your labeling script
(`{camera}_paths`, `{camera}_path_lengths`, `{camera}_path_timesteps`, and similarly for masks).

Example usage:

python examples/usc_widowx/convert_pathmask_uw_data_to_lerobot.py \
    --source-repo-id jesbu1/uw_widowx_8_8_lerobot \
    --paths-masks-file /Volumes/Sandisk\ 1TB/uw_widowx_labels_3b/uw_widowx_8_8_lerobot_paths_masks.h5 \
    --repo-id jesbu1/uw_widowx_8_8_pathmask_lerobot \
    --push-to-hub
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Literal
import warnings

import numpy as np
import torch
import tyro
import tqdm
import h5py
import cv2

try:
    # for older lerobot versions before 2.0.0
    from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
    OLD_LEROBOT = True
except ImportError:
    # newer lerobot versions use HF_LEROBOT_HOME instead of LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
    OLD_LEROBOT = False

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Path/Mask utilities adapted from `convert_pathmask_libero_data_to_lerobot_vlm_preds.py`
from vila_utils.utils.decode import add_path_2d_to_img_alt_fast, add_mask_2d_to_img
from vila_utils.utils.encode import scale_path, smooth_path_rdp


@dataclasses.dataclass(frozen=True)
class Args:
    source_repo_id: str
    repo_id: str
    paths_masks_file: str
    camera_key: str = "images0"
    path_line_size: int = 2
    mask_ratio_min: float = 0.08
    mask_ratio_max: float = 0.10
    push_to_hub: bool = False


def create_output_dataset_from_source(repo_id: str, ds_in: LeRobotDataset, camera_key: str) -> LeRobotDataset:
    # Copy non-video features and the selected camera video feature from the source dataset
    features: Dict[str, Dict[str, object]] = {}
    camera_feature_key = f"observation.images.{camera_key}"
    camera_feature = ds_in.features[camera_feature_key]

    for key, spec in ds_in.features.items():
        if spec.get("dtype") != "video":
            features[key] = spec
        elif key == camera_feature_key:
            features[key] = spec

    # Derive new camera names for overlays following the input naming
    # Try to preserve numeric suffix from camera_key (e.g., images0 -> path_image0)
    import re as _re
    m = _re.search(r"(\d+)$", camera_key)
    suffix = m.group(1) if m else ""
    path_cam_name = f"path_image{suffix}"
    masked_path_cam_name = f"masked_path_image{suffix}"

    # New video features with same shape/names as the input camera feature
    features[f"observation.images.{path_cam_name}"] = {
        "dtype": camera_feature["dtype"],
        "shape": tuple(camera_feature["shape"]),
        "names": list(camera_feature["names"]),
    }
    features[f"observation.images.{masked_path_cam_name}"] = {
        "dtype": camera_feature["dtype"],
        "shape": tuple(camera_feature["shape"]),
        "names": list(camera_feature["names"]),
    }

    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type="widowx",
        features=features,
        image_writer_threads=12,
        image_writer_processes=0,
        use_videos=True,
    )


def process_path_obs(sample_img: np.ndarray, path: np.ndarray, path_line_size: int = 2, apply_rdp: bool = True) -> np.ndarray:
    """Draw a normalized 2D path onto the provided image (HWC, uint8)."""
    height, width = sample_img.shape[:2]
    min_in, max_in = np.zeros(2), np.array([width, height])
    min_out, max_out = np.zeros(2), np.ones(2)
    path_scaled = scale_path(path, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
    if apply_rdp:
        path_scaled = smooth_path_rdp(path_scaled, tolerance=0.05)
    return add_path_2d_to_img_alt_fast(sample_img, path_scaled, line_size=path_line_size)


def process_mask_obs(sample_img: np.ndarray, mask_points: np.ndarray, mask_pixels: int = 25, scale_mask: bool = True, apply_rdp: bool = True) -> np.ndarray:
    """Apply a mask defined by 2D points to the image (HWC, uint8)."""
    if scale_mask:
        height, width = sample_img.shape[:2]
        min_in, max_in = np.zeros(2), np.array([width, height])
        min_out, max_out = np.zeros(2), np.ones(2)
        mask_points = scale_path(mask_points, min_in=min_out, max_in=max_out, min_out=min_in, max_out=max_in)
    if apply_rdp:
        mask_points = smooth_path_rdp(mask_points, tolerance=0.05)
    return add_mask_2d_to_img(sample_img, mask_points, mask_pixels=mask_pixels)


def convert_from_lerobot_with_paths_masks(
    source_repo_id: str,
    repo_id: str,
    *,
    paths_masks_file: str,
    camera_key: str = "images0",
    path_line_size: int = 2,
    mask_ratio_min: float = 0.08,
    mask_ratio_max: float = 0.10,
    push_to_hub: bool = False,
) -> None:
    # Input dataset
    ds_in = LeRobotDataset(source_repo_id)
    # Create output dataset using source features
    dataset_out = create_output_dataset_from_source(repo_id, ds_in, camera_key)

    with h5py.File(paths_masks_file, "r", swmr=True) as path_masks_h5:
        # Iterate episodes by index ranges
        for episode_idx in tqdm.tqdm(range(ds_in.num_episodes), desc="Processing episodes"):
            from_idx = ds_in.episode_data_index["from"][episode_idx].item()
            to_idx = ds_in.episode_data_index["to"][episode_idx].item()

            # Load per-episode path/mask arrays for the specified camera, if present
            ep_group = path_masks_h5.get(f"episode_{episode_idx}")
            path_data = path_lengths = path_timesteps = None
            mask_data = mask_lengths = mask_timesteps = None
            if ep_group is not None:
                path_data = ep_group.get(f"{camera_key}_paths")
                path_lengths = ep_group.get(f"{camera_key}_path_lengths")
                path_timesteps = ep_group.get(f"{camera_key}_path_timesteps")
                mask_data = ep_group.get(f"{camera_key}_masks")
                mask_lengths = ep_group.get(f"{camera_key}_mask_lengths")
                mask_timesteps = ep_group.get(f"{camera_key}_mask_timesteps")

                if path_data is not None:
                    path_data = path_data[()]
                if path_lengths is not None:
                    path_lengths = path_lengths[()]
                if path_timesteps is not None:
                    path_timesteps = path_timesteps[()]
                if mask_data is not None:
                    mask_data = mask_data[()]
                if mask_lengths is not None:
                    mask_lengths = mask_lengths[()]
                if mask_timesteps is not None:
                    mask_timesteps = mask_timesteps[()]

            # Per-episode state for overlay
            mask_ratio = float(np.random.uniform(mask_ratio_min, mask_ratio_max))
            current_path: Optional[np.ndarray] = None
            current_mask: Optional[np.ndarray] = None
            next_path_idx = 0
            next_mask_idx = 0

            # Prepare convenience names for new overlay keys
            import re as _re
            m = _re.search(r"(\d+)$", camera_key)
            suffix = m.group(1) if m else ""
            path_cam_name = f"path_image{suffix}"
            masked_path_cam_name = f"masked_path_image{suffix}"

            # Determine whether the input camera feature is CHW or HWC
            camera_feature = ds_in.features[f"observation.images.{camera_key}"]
            is_chw = (
                len(camera_feature.get("shape", [])) == 3
                and camera_feature["shape"][0] == 3
            )

            def hwc_to_chw_tensor(img_hwc_uint8: np.ndarray) -> torch.Tensor:
                t = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float() / 255.0
                return t

            for i, frame_idx in enumerate(range(from_idx, to_idx)):
                # Get base image from input dataset (CHW float [0,1] -> HWC uint8)
                try:
                    frame_in: Dict = ds_in[frame_idx]
                except Exception as e:
                    warnings.warn(f"Skipping frame {frame_idx} due to access error: {e}")
                    continue

                img_tensor: torch.Tensor = frame_in[f"observation.images.{camera_key}"]
                img_hwc = (img_tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)

                # Defaults
                base_img = img_hwc
                path_img = base_img
                masked_path_img = base_img

                # Update overlays if episode group present
                if ep_group is not None:
                    if (
                        path_data is not None
                        and path_lengths is not None
                        and path_timesteps is not None
                        and next_path_idx < len(path_timesteps)
                        and i == int(path_timesteps[next_path_idx])
                    ):
                        length = int(path_lengths[next_path_idx])
                        current_path = path_data[next_path_idx, :length].copy()
                        next_path_idx += 1

                    if (
                        mask_data is not None
                        and mask_lengths is not None
                        and mask_timesteps is not None
                        and next_mask_idx < len(mask_timesteps)
                        and i == int(mask_timesteps[next_mask_idx])
                    ):
                        length = int(mask_lengths[next_mask_idx])
                        current_mask = mask_data[next_mask_idx, :length].copy()
                        next_mask_idx += 1

                    if current_path is not None:
                        path_img = process_path_obs(base_img.copy(), current_path, path_line_size=path_line_size)
                        if current_mask is not None:
                            masked = process_mask_obs(
                                base_img.copy(),
                                current_mask,
                                mask_pixels=int(base_img.shape[0] * mask_ratio),
                                scale_mask=np.all(current_mask <= 1),
                                apply_rdp=True,
                            )
                            masked_path_img = process_path_obs(masked.copy(), current_path, path_line_size=path_line_size)
                        else:
                            masked_path_img = path_img
                    else:
                        if current_mask is not None:
                            masked_path_img = process_mask_obs(
                                base_img.copy(),
                                current_mask,
                                mask_pixels=int(base_img.shape[0] * mask_ratio),
                                scale_mask=np.all(current_mask <= 1),
                                apply_rdp=True,
                            )

                # Start with non-video keys copied from source frame
                frame_out: Dict[str, object] = {}
                for key, spec in ds_in.features.items():
                    if spec.get("dtype") != "video":
                        # Copy data from frame_in, converting torch tensors to numpy if needed
                        val = frame_in.get(key)
                        if isinstance(val, torch.Tensor):
                            val = val.detach().cpu().numpy()
                        frame_out[key] = val

                # Include the original image key for the selected camera, matching input format
                if is_chw:
                    frame_out[f"observation.images.{camera_key}"] = img_tensor
                else:
                    frame_out[f"observation.images.{camera_key}"] = base_img

                # Add overlays following input naming
                if is_chw:
                    frame_out[f"observation.images.{path_cam_name}"] = hwc_to_chw_tensor(path_img)
                    frame_out[f"observation.images.{masked_path_cam_name}"] = hwc_to_chw_tensor(masked_path_img)
                else:
                    frame_out[f"observation.images.{path_cam_name}"] = path_img
                    frame_out[f"observation.images.{masked_path_cam_name}"] = masked_path_img

                if OLD_LEROBOT:
                    dataset_out.add_frame(frame_out)
                else:
                    dataset_out.add_frame(frame_out, task=f"episode_{episode_idx}")

            if OLD_LEROBOT:
                dataset_out.save_episode(task=f"episode_{episode_idx}")
            else:
                dataset_out.save_episode()

    if OLD_LEROBOT:
        dataset_out.consolidate(run_compute_stats=False)

    if push_to_hub:
        dataset_out.push_to_hub(
            repo_id=repo_id,
            private=False,
            push_videos=True,
            upload_large_folder=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(convert_from_lerobot_with_paths_masks)


