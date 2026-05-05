#!/bin/bash
#SBATCH --job-name=hdepic-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hdepic_ablation_%j.out
#SBATCH --error=logs/hdepic_ablation_%j.err

set -e

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

module load gcc/11.3.0 ffmpeg/4.4.1-h264

export HF_HOME=/scratch/izar/cljordan/cache/huggingface

# Replacement mode: auxiliary frames only, no native video
echo "=== Replacement mode ==="
uv run --frozen python scripts/run_hdepic_ablation.py \
    --config configs/hdepic.yaml \
    --log-file logs/hdepic_ablation_${SLURM_JOB_ID}_replace.log \
    "${@}"

# Augmentation mode: native video + auxiliary frames
echo "=== Augmentation mode ==="
uv run --frozen python scripts/run_hdepic_ablation.py \
    --config configs/hdepic.yaml \
    --augment \
    --log-file logs/hdepic_ablation_${SLURM_JOB_ID}_augment.log \
    "${@}"

echo "Done."
