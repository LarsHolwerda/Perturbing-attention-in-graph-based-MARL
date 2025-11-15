#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --output=/scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/Simple_tag/logs/igappo_sweep_%j.out
#SBATCH --error=/scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/Simple_tag/logs/igappo_sweep_%j.err
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-8

echo "Starting sweep on $(hostname) at $(date)"

cd /scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/Simple_tag
