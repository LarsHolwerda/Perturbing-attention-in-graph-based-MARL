#!/bin/bash

#SBATCH --job-name=pgappo_baseline
#SBATCH --output=/scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/Simple_tag/logs/pgappo_baseline_%j.out
#SBATCH --error=/scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/Simple_tag/logs/pgappo_baseline_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

echo "Starting job on $(hostname) at $(date)"

# Load conda environment

cd /scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/Simple_tag

/scratch/7990537/conda/envs/simple_tag/bin/python main.py --exp-name "PGAPPO" \
               --algorithm "PGAPPO" \
               --env-id "simple_tag" \
               --use-cuda True \
               --number-of-workers 8 \
               --seed 0 \
               --track True \
               --wandb-project-name "Simple Tag" \
               --wandb-entity "lars-holwerda-utrecht-university" \
               --env-steps-per-batch 60000 \
               --n-iters 201 \
               --num-epochs 30 \
               --minibatch-size 4000 \
               --learning-rate 0.00005 \
               --max-grad-norm 5.0 \
               --clip-epsilon 0.2 \
               --shared-backbone True \
               --noise-scale 0.8 \
               --window-size 4000 \
               --perturb-attention-start-step 3000000 \
               --normal-training-period 820000 \
               --perturbation-period 180000 \
               --gamma 0.99 \
               --lmbda 1.0 \
               --entropy-eps 0.1 \
               --n-agents 3 \
               --n-good 1 \
               --agent-training-steps 3000000 \
               --n-adversaries 2 \
               --n-obstacles 2 \
               --continuous-actions False \
               --record-steps 120000 \
               --env-steps-to-analyze 500000

echo "Job finished at $(date)"