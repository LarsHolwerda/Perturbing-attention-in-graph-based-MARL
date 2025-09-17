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
        default="GAPPO",
        help="the name of this experiment")
    
    parser.add_argument(
        "--mappo", 
        type=bool,
        default=False,
        help="Whether to use MAPPO"
    )

    parser.add_argument(
        "--env-id", 
        type=str, 
        default="simple_tag",
        help="the id of the environment")
    
    # Device configuration
    parser.add_argument(
        "--use-cuda",
        type=lambda x: bool(strtobool(x)),
        default="False",
        help="Whether to use cuda"
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
        default=8000,
        help="Number of frames collected per training iteration"
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=1000,
        help="Number of sampling and training iterations"
    )

    # Training
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of optimization steps per training iteration"
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=1000,
        help="Size of mini-batches in each optimization step"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
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
        default=0.05,
        help="Entropy coefficient in PPO loss"
    )

    # Episode / Env
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=250,
        help="Max episode steps before done"
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
        type=bool,
        default=True,
        help="Are the actions discrete or continuous?"
    )

    # Recording
    parser.add_argument(
        "--record-steps",
        type=int,
        default=400000,
        help="Number of steps between video recordings"
    )

    args = parser.parse_args()

    # Computed fields
    args.total_env_steps = args.env_steps_per_batch * args.n_iters
    args.num_tag_envs = args.env_steps_per_batch // args.max_cycles
    args.device = torch.device(0) if (args.use_cuda and not is_fork) else torch.device("cpu")
    print(args.device)
    return args