#!/bin/bash

#SBATCH --job-name=mappo_baseline
#SBATCH --output=logs/mappo_baseline_%j.out
#SBATCH --error=logs/mappo_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G


echo "Starting job on $(hostname) at $(date)"

# Load conda environment
source ~/.bashrc
conda activate simple_tag

cd /scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/Simple_tag

python main.py --exp-name "MAPPO" \
               --algorithm "MAPPO" \
               --env-id "simple_tag" \
               --use-cuda False \
               --number-of-workers 16 \
               --seed 0 \
               --track True \
               --wandb-project-name "Simple Tag" \
               --wandb-entity "lars-holwerda-utrecht-university" \
               --env-steps-per-batch 60000 \
               --n-iters 200 \
               --num-epochs 45 \
               --minibatch-size 4000 \
               --learning-rate 0.00005 \
               --max-grad-norm 5.0 \
               --mappo True \
               --clip-epsilon 0.2 \
               --gamma 0.99 \
               --lmbda 0.9 \
               --entropy-eps 0.1 \
               --n-agents 3 \
               --n-good 1 \
               --agent-training-steps 3000000 \
               --n-adversaries 2 \
               --n-obstacles 2 \
               --continuous-actions False \
               --record-steps 100000