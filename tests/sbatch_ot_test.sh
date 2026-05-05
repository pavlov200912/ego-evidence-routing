#!/bin/bash
#SBATCH --job-name=ot-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ot_test_%j.out
#SBATCH --error=logs/ot_test_%j.err

set -e
mkdir -p logs

cd "$SLURM_SUBMIT_DIR"
module load gcc/11.3.0 ffmpeg/4.4.1-h264

export HF_HOME=/scratch/izar/cljordan/cache/huggingface

# Test 1: fine_grained_action_localization — query "spatula"
echo "=== Test 1: fine_grained_action_localization ==="
uv run --frozen python tests/test_object_tracking.py \
    --category fine_grained_action_localization \
    --question-idx 0 \
    --max-candidates 150 \
    --budget 8 \
    --out-dir results/ot_test/action

# Test 2: ingredient_ingredient_adding_localization — query "salt" or similar
echo "=== Test 2: ingredient_ingredient_adding_localization ==="
uv run --frozen python tests/test_object_tracking.py \
    --category ingredient_ingredient_adding_localization \
    --question-idx 0 \
    --max-candidates 150 \
    --budget 8 \
    --out-dir results/ot_test/ingredient

echo "Done."
