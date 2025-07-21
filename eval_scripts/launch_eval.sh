#!/bin/bash

# Default parameter values
EVAL_SET=${1:-10}  # Default to libero_10 if not specified (options: 10, spatial, object, goal)
USE_PATH=${2:-0}   # Default to not using path
USE_MASK=${3:-0}   # Default to not using mask
VILA_GPU_ID=${4:-0} # Default to GPU 0 for VILA server
POLICY_GPU_ID=${5:-0} # Default to GPU 0 for Policy server
VILA_PORT=${6:-8000} # Default VILA port
POLICY_PORT=${7:-8001} # Default Policy port


VLM_QUERY_FREQUENCY=5 # how many times to call the VLM per action chunk

source ~/.bashrc

cd /home1/jessez/nvidia/VILA
echo "Running VILA server on GPU $VILA_GPU_ID (Port: $VILA_PORT - Note: Port used by client connection, server might default to 8000)"
# Use specified GPU for VILA server - Assuming VILA server runs on default port 8000, client needs to connect to VILA_PORT
conda run -n vila --no-capture-output /bin/bash -c "CUDA_VISIBLE_DEVICES=$VILA_GPU_ID python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &

# Wait for the model to load
sleep 30

# Base command for policy server, now using POLICY_PORT
SERVE_CMD_BASE="uv run scripts/serve_policy.py --port $POLICY_PORT policy:checkpoint"

# Append policy specific arguments based on USE_PATH and USE_MASK
if [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "0" ]; then
    #SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_path --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path/pi0_libero_90_path_bs164_rdp/35000/"
    SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_path --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path/pi0_libero_low_mem_finetune_path_new/30000/"
elif [ "$USE_PATH" = "0" ] && [ "$USE_MASK" = "1" ]; then
    SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_masked --policy.dir=checkpoints/pi0_libero_low_mem_finetune_masked/pi0_libero_90_masked_bs164_rdp/30000/"
elif [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "1" ]; then
    #SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_path_masked --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path_masked/pi0_libero_90_path_masked_bs164_rdp/30000/"
    SERVE_CMD_POLICY_ARGS="--policy.config=pi0_libero_low_mem_finetune_path_masked --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path_masked/pi0_libero_low_mem_finetune_path_masked_mask0.01_0.12/30000/"
else
    SERVE_CMD_POLICY_ARGS="" # No specific policy args if neither or only base is used
fi

# Construct the full serving command with CUDA_VISIBLE_DEVICES
SERVE_CMD="CUDA_VISIBLE_DEVICES=$POLICY_GPU_ID $SERVE_CMD_BASE $SERVE_CMD_POLICY_ARGS"


cd /home1/jessez/nvidia/openpi

echo "Running policy server on GPU $POLICY_GPU_ID, Port $POLICY_PORT: $SERVE_CMD"
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

# Build command with optional flags, using VILA_PORT and POLICY_PORT
EVAL_CMD="python examples/libero/main.py --args.task_suite_name=$TASK_SUITE --args.vlm_server_ip=http://0.0.0.0:$VILA_PORT --args.port $POLICY_PORT --args.vlm_query_frequency $VLM_QUERY_FREQUENCY"

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
echo "Running evaluation: $EVAL_CMD"
$EVAL_CMD 