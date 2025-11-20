import argparse
from distutils.util import strtobool
import multiprocessing
import torch

# Devices
is_fork = multiprocessing.get_start_method() == "fork"

def parse_training_args():
    parser = argparse.ArgumentParser()

    # General information on the experiment for logging purposes
    parser.add_argument(
        "--exp-name", 
        type=str, 
        default="PIGAPPO",
        help="the name of this experiment")

    parser.add_argument(
        "--algorithm", 
        type=str, 
        default="PIGAPPO",
        help="the algorithm which will be trained")

    parser.add_argument(
        "--env-id", 
        type=str, 
        default="simple_tag",
        help="the id of the environment")
    
    # Device configuration
    parser.add_argument(
        "--use-cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="Whether to use cuda"
    )

    parser.add_argument(
        "--number-of-workers",
        type=int,
        default=2,
        help="Number of parallel workers"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    # Logging
    parser.add_argument(
        "--track", 
        type=lambda x: bool(strtobool(x)), 
        default=True,
        help="if toggled, this experiment will be tracked with Weights and Biases"
    ) 
    parser.add_argument(
        "--wandb-project-name", 
        type=str, 
        default="Simple Tag",
        help="the wandb's project name"
    )
    parser.add_argument(
        "--wandb-entity", 
        type=str, 
        default="lars-holwerda-utrecht-university",
        help="the entity (team) of wandb's project"
    )

    # Sampling
    parser.add_argument(
        "--env-steps-per-batch",
        type=int,
        default=250,
        help="Number of frames collected per training iteration"
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=4,
        help="Number of sampling and training iterations"
    )

    # Training
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of optimization steps per training iteration"
    )
    parser.add_argument(
        "--minibatch-size",
        "--minibatch_size",
        type=int,
        default=250,
        help="Size of mini-batches in each optimization step"
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=5.0,
        help="Maximum norm for gradients"
    )

    # PPO
    parser.add_argument(
        "--mappo", 
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use MAPPO"
    )

    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
        help="Clip value for PPO loss"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--lmbda",
        type=float,
        default=0.95,
        help="Lambda for generalized advantage estimation"
    )
    parser.add_argument(
        "--entropy-eps",
        type=float,
        default=0.01,
        help="Entropy coefficient in PPO loss"
    )

    # Gappo
    parser.add_argument(
        "--shared-backbone", 
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to use a shared backbone for actor and critic in GAPPO"
    )

    # PGAPPO / PIGAPPO
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.1,
        help="Scale of the noise to perturb attention logits"
    )
    
    parser.add_argument(
        "--window-size",
        type=int,
        default=30,
        help="Size of the window for pre-computed noise for attention perturbation"
    )

    parser.add_argument(
        "--perturb-attention-start-step",
        type=int,
        default=250,
        help="Number of steps after which to start perturbing attention weights"
    )

    parser.add_argument(
        "--normal-training-period",
        type=int,
        default=0,
        help="Number of steps between perturbation periods, to stabilizes training"
    )

    parser.add_argument(
        "--perturbation-period",
        type=int,
        default=1,
        help="Number of steps during which to perturb attention weights"
    )

    # Episode / Env
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=250,
        help="Max episode steps before done"
    )

    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
        help="Number of agents in the environment"
    )

    parser.add_argument(
        "--n-good",
        type=int,
        default=1,
        help="Number of good agents in the environment"
    )

    parser.add_argument(
        "--agent-training-steps",
        type=int,
        default=3000000,
        help="Number of training steps to train the good agent"
    )

    parser.add_argument(
        "--n-adversaries",
        type=int,
        default=2,
        help="Number of adversaries in the environment"
    )

    parser.add_argument(
        "--n-obstacles",
        type=int,
        default=2,
        help="Number of obstacles in the environment"
    )

    parser.add_argument(
        "--continuous-actions",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Are the actions discrete or continuous?"
    )

    # Recording
    parser.add_argument(
        "--record-steps",
        type=int,
        default=40000,
        help="Number of steps between video recordings"
    )

    parser.add_argument(
        "--env-steps-to-analyze",
        type=int,
        default=50000,
        help="Number of environment steps to store for analysis at the end of training"
    )

    args = parser.parse_args()

    # Computed fields
    args.total_env_steps = args.env_steps_per_batch * args.n_iters
    args.num_tag_envs = args.env_steps_per_batch // args.max_cycles
    args.device = torch.device(0) if (args.use_cuda and not is_fork) else torch.device("cpu")
    print(args.device)
    return args