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
        default="IPPO",
        help="the name of this experiment")
    
    parser.add_argument(
        "--env-id", 
        type=str, 
        default="academy_3_v_1_with_keeper",
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
        default="Google Research Football",
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
        default=4000,
        help="Number of frames collected per training iteration"
    )
    parser.add_argument(
        "--n-iters",
        type=int,
        default=800,
        help="Number of sampling and training iterations"
    )

    # Training
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        help="Number of optimization steps per training iteration"
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=500,
        help="Size of mini-batches in each optimization step"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=10.0,
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
        default=0.0,
        help="Entropy coefficient in PPO loss"
    )

    # Episode / Env
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Max episode steps before done"
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=3,
        help="Number of agents in the environment"
    )


    args = parser.parse_args()

    # Computed fields
    args.total_env_steps = args.env_steps_per_batch * args.n_iters
    args.num_grf_envs = args.env_steps_per_batch // args.max_steps
    args.device = torch.device(0) if (args.use_cuda and not is_fork) else torch.device("cpu")
    print(args.device)
    return args