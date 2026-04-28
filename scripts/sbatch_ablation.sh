#!/bin/bash
#SBATCH --job-name=eer-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

set -e

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

export HF_HOME=/scratch/izar/cljordan/cache/huggingface

uv run python scripts/run_egotextvqa_ablation.py \
    --config configs/egotextvqa.yaml \
    "${@}"
