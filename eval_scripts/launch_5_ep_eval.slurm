#!/bin/bash
#SBATCH --account=jessetho_1023
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

POLICY_GPU_ID=0
VILA_GPU_ID=0

VLM_QUERY_FREQUENCY=5 # how many times to call the VLM per action chunk

source ~/.bashrc

cd /home1/jessez/nvidia/VILA
echo "Running VILA server"
conda run -n vila --no-capture-output /bin/bash -c "CUDA_VISIBLE_DEVICES=$VILA_GPU_ID python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &

# Wait for the model to load
sleep 60

cd /home1/jessez/nvidia/openpi

# Base command for policy server
SERVE_CMD_BASE="uv run scripts/serve_policy.py --port 8001 policy:checkpoint"

# Append policy specific arguments based on USE_PATH and USE_MASK
if [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "0" ]; then
    SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_path --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path/pi0_libero_low_mem_finetune_path_new_maxep5/10000/"
elif [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "1" ]; then
    SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_path_masked --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path_masked/pi0_libero_low_mem_finetune_path_masked_mask_maxep5/10000/"
else
  SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune --policy.dir=checkpoints/pi0_libero_low_mem_finetune/pi0_libero_test_5ep/30000"
fi

# Construct the full serving command with CUDA_VISIBLE_DEVICES
SERVE_CMD="CUDA_VISIBLE_DEVICES=$POLICY_GPU_ID $SERVE_CMD_BASE $SERVE_CMD_POLICY_ARGS"
echo "Running serving command: $SERVE_CMD"
bash -c "$SERVE_CMD" &

sleep 30

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


# Build command with optional flags
EVAL_CMD="python examples/libero/main.py --args.task_suite_name=$TASK_SUITE --args.vlm_server_ip=http://0.0.0.0:8000 --args.port 8001 --args.vlm_query_frequency=$VLM_QUERY_FREQUENCY --args.wandb_name_suffix=-vlmfreq${VLM_QUERY_FREQUENCY}_test10ep --args.wandb_group_prefix=eval-pi0-test10ep_7/26"

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