#!/bin/bash
#SBATCH --job-name=asset-pricing
#SBATCH --output=/scratch/network/lo8603/final_project/outputs/logs/%j.out
#SBATCH --error=/scratch/network/lo8603/final_project/outputs/logs/%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

mkdir -p /scratch/network/lo8603/final_project/outputs/logs

echo "Shell started on $(hostname) at $(date)"

cd /scratch/network/lo8603/final_project

echo "Starting Python..."

/scratch/network/lo8603/thesis/conda/envs/myenv/bin/python -u run.py --processed data/processed/panel_processed.parquet --run-extensions --run-backtest

echo "Python exited with code $?"

