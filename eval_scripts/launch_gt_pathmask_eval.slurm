#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=185G #185G to get ampere or newer gpus
#SBATCH --partition=gpu
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --gres=gpu:1

# Default parameter values
EVAL_SET=${1:-10}  # Default to libero_10 if not specified (options: 10, spatial, object, goal)
USE_PATH=${2:-0}   # Default to not using path
USE_MASK=${3:-0}   # Default to not using mask
NO_PROPRIO=${4:-0} # Default to not using proprio

VLM_QUERY_FREQUENCY=10 # how many times to call the VLM per action chunk

source ~/.bashrc

cd /home1/jessez/nvidia/openpi

# Base command for policy server
SERVE_CMD_BASE="uv run scripts/serve_policy.py --port 8000 policy:checkpoint"

# Append policy specific arguments based on USE_PATH and USE_MASK
if [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "0" ]; then
    SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_fullpath_path --policy.dir=checkpoints/pi0_libero_low_mem_finetune_fullpath_path/pi0_libero_90_fullpath_path_bs148_rdp/30000/"
    if [ "$NO_PROPRIO" = "1" ]; then
        SERVE_CMD_POLICY_ARGS="$SERVE_CMD_POLICY_ARGS --policy.config=pi0_libero_low_mem_finetune_fullpath_path_no_proprio --policy.dir=checkpoints/pi0_libero_low_mem_finetune_fullpath_path_no_proprio/pi0_libero_90_fullpath_path_bs148_rdp_no_proprio/30000"
    fi
elif [ "$USE_PATH" = "0" ] && [ "$USE_MASK" = "1" ]; then
    SERVE_CMD_POLICY_ARGS="" # TODO: haven't implemented it yet
elif [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "1" ]; then
    SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_fullpath_path_masked --policy.dir=checkpoints/pi0_libero_low_mem_finetune_fullpath_path_masked/pi0_libero_90_fullpath_path_masked_bs148_rdp/30000/"
    if [ "$NO_PROPRIO" = "1" ]; then
        SERVE_CMD_POLICY_ARGS="$SERVE_CMD_POLICY_ARGS --policy.config=pi0_libero_low_mem_finetune_fullpath_path_masked_no_proprio --policy.dir=checkpoints/pi0_libero_low_mem_finetune_fullpath_path_masked_no_proprio/pi0_libero_90_fullpath_path_masked_bs148_rdp_no_proprio/30000"
    fi
else
  SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_no_proprio  --policy.dir=checkpoints/pi0_libero_low_mem_finetune_no_proprio/pi0_libero_90_fullpath_path_no_proprio_bs148_rdp/30000"
fi

# Construct the full serving command with CUDA_VISIBLE_DEVICES
SERVE_CMD="CUDA_VISIBLE_DEVICES=$POLICY_GPU_ID $SERVE_CMD_BASE $SERVE_CMD_POLICY_ARGS"
echo "Running serving command: $SERVE_CMD"
bash -c "$SERVE_CMD" &

sleep 20

# Determine task suite based on EVAL_SET
if [ "$EVAL_SET" = "10" ]; then
  TASK_SUITE="libero_10"
elif [ "$EVAL_SET" = "spatial" ]; then
  TASK_SUITE="libero_spatial"
elif [ "$EVAL_SET" = "object" ]; then
  TASK_SUITE="libero_object"
elif [ "$EVAL_SET" = "goal" ]; then
  TASK_SUITE="libero_goal"
else
  echo "Invalid EVAL_SET: $EVAL_SET. Using default libero_10."
  TASK_SUITE="libero_10"
fi


path_and_mask_file_dir="/home1/jessez/scratch_data/libero_processed_256_05_12/masked_vla_data"
# Build command with optional flags
EVAL_CMD="python examples/libero/main.py --args.task_suite_name=$TASK_SUITE --args.use_ground_truth_path_masks --args.path_and_mask_file_dir=$path_and_mask_file_dir"

# Add draw_path if USE_PATH is 1
if [ "$USE_PATH" = "1" ]; then
  EVAL_CMD="$EVAL_CMD --args.draw_path"
fi

# Add draw_mask if USE_MASK is 1
if [ "$USE_MASK" = "1" ]; then
  EVAL_CMD="$EVAL_CMD --args.draw_mask"
fi

# Execute the command
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
echo "Running: $EVAL_CMD"
$EVAL_CMD