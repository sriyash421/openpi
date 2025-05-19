# Convert Bridge Data to LeRobot Format

This script converts Bridge data into LeRobot dataset format, with support for paths and masks.

## Features

- Converts Bridge data to LeRobot v2.0 format
- Supports multiple camera views
- Processes and includes path visualizations
- Processes and includes mask visualizations
- Combines paths and masks for each camera view

## Usage

Basic usage:
```bash
uv run examples/bridge/convert_bridge_data_to_lerobot.py --data_dir /path/to/data --repo_id bridge_v2_lerobot
```

### Optional Arguments

- `--push-to-hub`: Push the dataset to Hugging Face Hub
- `--dataset-config`: Configure dataset parameters:
  - `image_height`: Image height (default: 256)
  - `image_width`: Image width (default: 256)
  - `path_line_size`: Line size for path visualization (default: 3)
  - `mask_pixels`: Size of mask pixels (default: 25)
  - `image_writer_processes`: Number of image writer processes (default: 10)
  - `image_writer_threads`: Number of image writer threads (default: 5)

Example with custom configuration:
```bash
uv run examples/bridge/convert_bridge_data_to_lerobot.py \
    --data_dir /path/to/data \
    --repo_id bridge_v2_lerobot \
    --dataset-config.image_height 320 \
    --dataset-config.image_width 320 \
    --dataset-config.path_line_size 4 \
    --dataset-config.mask_pixels 30 \
    --push-to-hub
```

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

