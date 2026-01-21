#!/bin/bash
#SBATCH --job-name=pi0_libero_90_low_mem_finetune
#SBATCH --qos=normal
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/projects/weirdlab/sriyash/slurm_logs/slurm-%x-%j.out

cd /gpfs/projects/weirdlab/sriyash/openpi
source .venv/bin/activate

export SCRUBBED_PATH="/gpfs/scrubbed/sriyash"
export UV_CACHE_DIR="$SCRUBBED_PATH/.cache/uv"
export OPENPI_DATA_HOME=$SCRUBBED_PATH
export HF_HOME="$SCRUBBED_PATH/huggingface"
export HF_DATASETS_CACHE="$SCRUBBED_PATH/hf_datasets_cache"

# exp name with date time
exp_name="pi0_libero_90_low_mem_finetune_$(date +'%Y%m%d_%H%M%S')"

#python scripts/compute_norm_stats.py --config-name pi0_libero_90_low_mem_finetune

python scripts/train.py pi0_libero_90_low_mem_finetune --exp-name=$exp_name --overwrite
