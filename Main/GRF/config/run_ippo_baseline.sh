#!/bin/bash

#SBATCH --job-name=ippo_baseline
#SBATCH --output=/scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/GRF/logs/ippo_baseline_%j.out
#SBATCH --error=/scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/GRF/logs/ippo_baseline_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

echo "Starting job on $(hostname) at $(date)"

# Load conda environment

cd /scratch/7990537/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main/GRF

/scratch/7990537/conda/envs/grf_env/bin/python main.py --exp-name "IPPO" \
               --algorithm "IPPO" \
               --env-id "academy_3_vs_1_with_keeper" \
               --use-cuda True \
               --number-of-workers 8 \
               --seed 0 \
               --track True \
               --wandb-project-name "Google Research Football" \
               --wandb-entity "lars-holwerda-utrecht-university" \
               --env-steps-per-batch 9600 \
               --n-iters 651 \
               --num-epochs 45 \
               --minibatch-size 3200 \
               --learning-rate 0.0005 \
               --max-grad-norm 5.0 \
               --mappo False \
               --clip-epsilon 0.2 \
               --gamma 0.99 \
               --lmbda 0.95 \
               --entropy-eps 0.0 \
               --n-agents 3 \
               --record-steps 124750 \
               --num-episodes-to-record 1 \
               --policy-iterations 10 \
               --obs-dim 58

echo "Job finished at $(date)"