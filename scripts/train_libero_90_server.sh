#!/bin/bash

# Configuration
NUM_RUNS=20
EXP_NAME="pi0_libero_90_LoRA_finetune_8gpu"
LOG_DIR="/home/jeszhang/data/openpi/slurm_outputs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

# Log file
LOG_FILE="$LOG_DIR/simpler_processing_$(date +%Y%m%d_%H%M%S).log"

echo "$(date): Starting processing with $NUM_RUNS iterations" | tee -a $LOG_FILE

# Loop to run the command multiple times
for ((i=1; i<=NUM_RUNS; i++)); do
    echo "$(date): Starting run $i of $NUM_RUNS" | tee -a $LOG_FILE
    
    # Run the command with srun and a timeout to ensure it doesn't run too long
    # Using 150 minutes (2.5 hours) as the timeout for each run
    srun --account=nvr_srl_simpler \
                    --time=4:00:00 \
                    --gpus=8 \
                    --ntasks=1 \
                    --exclusive \
                    --partition=grizzly,polar,polar3,polar4,batch_singlenode,backfill_block1 \
                    /bin/bash -c "cd /home/jeszhang/data/openpi && \ 
                    source activate .venv/bin/activate && \
                    export WANDB_API_KEY=41495c354f793dc48bd32583a0e3c653f9991221 && \
                    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune_8gpu --exp-name=$EXP_NAME --resume" 2>&1 | tee -a $LOG_FILE
    
    EXIT_STATUS=$?
    
    # Log the exit status
    echo "$(date): Run $i completed with exit status $EXIT_STATUS" | tee -a $LOG_FILE
    
    # Optional: Add a short sleep between runs to allow system to stabilize
    sleep 5
done

echo "$(date): All $NUM_RUNS runs completed. Processing finished." | tee -a $LOG_FILE 