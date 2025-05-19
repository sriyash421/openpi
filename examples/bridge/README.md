# Convert Bridge Data to LeRobot Format

This repository contains scripts to convert Bridge data into LeRobot dataset format, with support for paths and masks.

## Overview

The conversion process happens in two steps:
1. Generate paths and masks using VLM inference
2. Convert the data to LeRobot format with the generated paths and masks

## Step 1: Generate Paths and Masks

First, generate paths and masks using VLM inference:

```bash
uv run examples/bridge/generate_paths_masks.py \
    --data_dir /path/to/data \
    --output_dir /path/to/output \
    --vlm_server_ip http://your-vlm-server:8000 \
    --draw_path True \
    --draw_mask True
```

### Optional Arguments for Path/Mask Generation

- `--vlm_server_ip`: VLM server address (default: "http://0.0.0.0:8000")
- `--resize_size`: Size to resize images for VLM (default: 224)
- `--draw_path`: Whether to generate paths (default: True)
- `--draw_mask`: Whether to generate masks (default: True)
- `--flip_image_horizontally`: Whether to flip images horizontally (default: False)

## Step 2: Convert to LeRobot Format

Then, convert the data to LeRobot format using the generated paths and masks:

```bash
uv run examples/bridge/convert_bridge_data_to_lerobot.py \
    --data_dir /path/to/data \
    --repo_id bridge_v2_lerobot \
    --paths_masks_file /path/to/output/bridge_paths_masks.h5 \
    --dataset-config.image_height 256 \
    --dataset-config.image_width 256 \
    --dataset-config.path_line_size 3 \
    --dataset-config.mask_pixels 25 \
    --push-to-hub
```

### Required Arguments for Conversion

- `--data_dir`: Directory containing the Bridge dataset
- `--repo_id`: Repository ID for the LeRobot dataset
- `--paths_masks_file`: Path to the HDF5 file containing generated paths and masks

### Optional Arguments for Conversion

- `--push-to-hub`: Push the dataset to Hugging Face Hub
- `--dataset-config`: Configure dataset parameters:
  - `image_height`: Image height (default: 256)
  - `image_width`: Image width (default: 256)
  - `path_line_size`: Line size for path visualization (default: 3)
  - `mask_pixels`: Size of mask pixels (default: 25)
  - `image_writer_processes`: Number of image writer processes (default: 10)
  - `image_writer_threads`: Number of image writer threads (default: 5)

## Output Format

The converted dataset includes:
- Original camera views
- Path visualizations for each camera view
- Mask visualizations for each camera view
- Combined path and mask visualizations for each camera view
- Robot state and action data

Each camera view (image_0, image_1, image_2, image_3) will have its own:
- Path visualization
- Mask visualization
- Combined path and mask visualization

```bash
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /home1/jessez/tensorflow_datasets/ --repo_id bridge_v2_lerobot
```

