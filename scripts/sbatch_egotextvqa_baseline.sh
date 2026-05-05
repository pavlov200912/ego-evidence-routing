#!/bin/bash
#SBATCH --job-name=eer-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

set -e

mkdir -p logs

cd "$SLURM_SUBMIT_DIR"

export HF_HOME=/scratch/izar/cljordan/cache/huggingface

uv run python scripts/run_egotextvqa_baseline.py \
    --config configs/egotextvqa.yaml \
    "${@}"
