#!/bin/bash
#SBATCH --job-name=build-chars
#SBATCH --output=/scratch/network/lo8603/final_project/outputs/logs/%j.out
#SBATCH --error=/scratch/network/lo8603/final_project/outputs/logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

mkdir -p /scratch/network/lo8603/final_project/outputs/logs

echo "Started on $(hostname) at $(date)"

cd /scratch/network/lo8603/final_project

# WRDS credentials — fill in your password before submitting
export WRDS_USER="lo8603"
export WRDS_PASS="FranzLisztCorgi77"

/scratch/network/lo8603/thesis/conda/envs/myenv/bin/python -u src/build_missing.py

echo "Exited with code $? at $(date)"