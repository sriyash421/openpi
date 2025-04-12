# Instructions for training on LIBERO:
First follow the repo install instructions.
Setup UV in a virtual environment in this repo before uv installing things.
Make sure conda is deactivated!!!!!
```bash
uv venv
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync
```
Then:
```bash
uv pip install tensorflow tensorflow_datasets shapely
```

Follow the instructions in my openvla repo to install and generate the modified LIBERO dataset: [here](https://github.com/jesbu1/openvla).

There are two ways to convert the LIBERO dataset to a LeRobot dataset:

### Option 1: Basic Conversion
Convert the basic modified LIBERO dataset to a LeRobot dataset:
```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /home/jeszhang/tensorflow_datasets/
```

### Option 2: Conversion with Path Masks and Subtasks
For more advanced training with path masks and subtask instructions, you can use the following script. note that the data_dir expects the data dir of the OpenVLA processed hdf5 LIBERO data, not the tensorflow dataset. Download from [here](https://huggingface.co/datasets/jesbu1/libero_90_openvla_processed).
```bash
uv run examples/libero/convert_pathmask_libero_data_to_lerobot.py --data_dir /home1/jessez/scratch_data/libero_openvla_processed_datasets --path_and_mask_file_dir /home1/jessez/project_data/libero_90_processed_256 
```

This script supports several additional options:
- `--use_subtask_instructions`: Divides episodes by subtask instructions instead of using full task instructions
- `--push_to_hub`: Pushes the processed dataset to Hugging Face Hub

The path mask conversion relies on annotation files containing masks, paths, and subtask information.

I've already processed the training stats and norms in `assets/pi0_libero_low_mem_finetune/jesbu1/libero_90_lerobot/norm_stats.json`. If not running `pi0_libero_low_mem_finetune`, you can copy the `norm_stats.json` file to the `assets/CONFIG_NAME/jesbu1/libero_90_lerobot/` directory.

## Training π₀ with LoRA

You may need to change the `repo_id` in the `src/openpi/training/config.py` file for the `pi0_libero_low_mem_finetune` config to the `jesbu1/libero_90_lerobot` dataset and where it is on your machine. It should by default be in `/home/$USER/.cache/huggingface/lerobot/jesbu1/libero_90_lerobot/`. You can also change the `local_files_only` flag to `False` in the `src/openpi/training/config.py` file to use the local dataset.

### Basic Training
Then train:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=EXP_NAME --overwrite
```

### Training with Validation
To train with validation, you can specify a validation dataset in the config. The validation dataset should be a separate LeRobot dataset that follows the same format as your training data. The validation will run every `validation_interval` steps (default 1000) and log the validation metrics to wandb.

Example config modification:
```python
config = TrainConfig(
    # ... other config ...
    validation_data=LeRobotLiberoDataConfig(
        repo_id="jesbu1/libero_90_lerobot_val",  # Your validation dataset
        base_config=DataConfig(
            local_files_only=True,
            prompt_from_task=True,
        ),
    ),
    validation_interval=1000,  # Run validation every 1000 steps
)
```

### Training with Path Masks
You can also train with path masks by running the following:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_path --exp-name=EXP_NAME --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_masked --exp-name=EXP_NAME --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_path_masked --exp-name=EXP_NAME --overwrite

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_path_no_proprio --exp-name=EXP_NAME --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_masked_no_proprio --exp-name=EXP_NAME --overwrite
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_path_masked_no_proprio --exp-name=EXP_NAME --overwrite
```

## Evaluation of LIBERO
Once done training, you can evaluate the model by running the following command to initialize a policy server:
```bash
CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero_low_mem_finetune --policy.dir=checkpoints/pi0_libero_90_LoRA_finetune_8gpu/29999/ 
```

In a separate terminal, run the following command to run the Libero evaluation script:
```bash
# Create virtual environment
conda deactivate
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
uv pip install wandb
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py --args.task_suite_name=libero_10
python examples/libero/main.py --args.task_suite_name=libero_spatial
python examples/libero/main.py --args.task_suite_name=libero_object
python examples/libero/main.py --args.task_suite_name=libero_goal
```

# openpi

openpi holds open-source models and packages for robotics, published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

Currently, this repo contains two types of models:
- the [π₀ model](https://www.physicalintelligence.company/blog/pi0), a flow-based diffusion vision-language-action model (VLA)
- the [π₀-FAST model](https://www.physicalintelligence.company/research/fast), an autoregressive VLA, based on the FAST action tokenizer.

For both models, we provide _base model_ checkpoints, pre-trained on 10k+ hours of robot data, and examples for using them out of the box or fine-tuning them to your own datasets.

This is an experiment: $\pi_0$ was developed for our own robots, which differ from the widely used platforms such as [ALOHA](https://tonyzhaozh.github.io/aloha/) and [DROID](https://droid-dataset.github.io/), and though we are optimistic that researchers and practitioners will be able to run creative new experiments adapting $\pi_0$ to their own platforms, we do not expect every such attempt to be successful. All this is to say: $\pi_0$ may or may not work for you, but you are welcome to try it and see!


## Requirements

To run the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Inference          | > 8 GB          | RTX 4090           |
| Fine-Tuning (LoRA) | > 22.5 GB       | RTX 4090           |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |

The repo has been tested with Ubuntu 22.04, we do not currently support other operating systems.

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
```

NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

**Docker**: As an alternative to uv installation, we provide instructions for installing openpi using Docker. If you encounter issues with your system setup, consider using Docker to simplify installation. See [Docker Setup](docs/docker.md) for more details.




## Model Checkpoints

### Base Models
We provide multiple base VLA model checkpoints. These checkpoints have been pre-trained on 10k+ hours of robot data, and can be used for fine-tuning.

| Model        | Use Case    | Description                                                                                                 | Checkpoint Path                                |
| ------------ | ----------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| $\pi_0$      | Fine-Tuning | Base diffusion [π₀ model](https://www.physicalintelligence.company/blog/pi0) for fine-tuning                | `s3://openpi-assets/checkpoints/pi0_base`      |
| $\pi_0$-FAST | Fine-Tuning | Base autoregressive [π₀-FAST model](https://www.physicalintelligence.company/research/fast) for fine-tuning | `s3://openpi-assets/checkpoints/pi0_fast_base` |

### Fine-Tuned Models
We also provide "expert" checkpoints for various robot platforms and tasks. These models are fine-tuned from the base models above and intended to run directly on the target robot. These may or may not work on your particular robot. Since these checkpoints were fine-tuned on relatively small datasets collected with more widely available robots, such as ALOHA and the DROID Franka setup, they might not generalize to your particular setup, though we found some of these, especially the DROID checkpoint, to generalize quite broadly in practice.

| Model                    | Use Case  | Description                                                                                                                                                                                              | Checkpoint Path                                       |
| ------------------------ | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| $\pi_0$-FAST-DROID       | Inference | $\pi_0$-FAST model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/), can perform a wide range of simple table-top manipulation tasks 0-shot in new scenes on the DROID robot platform | `s3://openpi-assets/checkpoints/pi0_fast_droid`       |
| $\pi_0$-DROID            | Fine-Tuning | $\pi_0$ model fine-tuned on the [DROID dataset](https://droid-dataset.github.io/), faster inference than $\pi_0$-FAST-DROID, but may not follow language commands as well | `s3://openpi-assets/checkpoints/pi0_droid` |
| $\pi_0$-ALOHA-towel      | Inference | $\pi_0$ model fine-tuned on internal ALOHA data, can fold diverse towels 0-shot on [ALOHA](https://tonyzhaozh.github.io/aloha/) robot platforms                                                          | `s3://openpi-assets/checkpoints/pi0_aloha_towel`      |
| $\pi_0$-ALOHA-tupperware | Inference | $\pi_0$ model fine-tuned on internal ALOHA data, can unpack food from a tupperware container                                                                                                             | `s3://openpi-assets/checkpoints/pi0_aloha_tupperware` |
| $\pi_0$-ALOHA-pen-uncap  | Inference | $\pi_0$ model fine-tuned on [public ALOHA data](https://dit-policy.github.io/), can uncap a pen                                                                                                                                    | `s3://openpi-assets/checkpoints/pi0_aloha_pen_uncap`  |


By default, checkpoints are automatically downloaded from `s3://openpi-assets` and are cached in `~/.cache/openpi` when needed. You can overwrite the download path by setting the `OPENPI_DATA_HOME` environment variable.




## Running Inference for a Pre-Trained Model

Our pre-trained model checkpoints can be run with a few lines of code (here our $\pi_0$-FAST-DROID model):
```python
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]
```
You can also test this out in the [example notebook](examples/inference.ipynb).

We provide detailed step-by-step examples for running inference of our pre-trained checkpoints on [DROID](examples/droid/README.md) and [ALOHA](examples/aloha_real/README.md) robots.

**Remote Inference**: We provide [examples and code](docs/remote_inference.md) for running inference of our models **remotely**: the model can run on a different server and stream actions to the robot via a websocket connection. This makes it easy to use more powerful GPUs off-robot and keep robot and policy environments separate.

**Test inference without a robot**: We provide a [script](examples/simple_client/README.md) for testing inference without a robot. This script will generate a random observation and run inference with the model. See [here](examples/simple_client/README.md) for more details.





## Fine-Tuning Base Models on Your Own Data

We will fine-tune the $\pi_0$-FAST model on the [Libero dataset](https://libero-project.github.io/datasets) as a running example for how to fine-tune a base model on your own data. We will explain three steps:
1. Convert your data to a LeRobot dataset (which we use for training)
2. Defining training configs and running training
3. Spinning up a policy server and running inference

### 1. Convert your data to a LeRobot dataset

We provide a minimal example script for converting Libero data to a LeRobot dataset in [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py). You can easily modify it to convert your own data! You can download the raw Libero dataset from [here](https://huggingface.co/datasets/openvla/modified_libero_rlds), and run the script with:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

### 2. Defining training configs and running training

To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for Libero below, which you can modify for your own dataset:

- [`LiberoInputs` and `LiberoOutputs`](src/openpi/policies/libero_policy.py): Defines the data mapping from the Libero environment to the model and vice versa. Will be used for both, training and inference.
- [`