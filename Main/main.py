import multiprocessing as mp
mp.set_start_method("spawn", force=True)
import torch
from tensordict.nn import set_composite_lp_aggregate
from config import parse_training_args
from env import create_env, create_render_env
from ippo import IPPO

# Set seed
torch.manual_seed(0)
# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

# Get command line arguments
args = parse_training_args()

# Log to wandb
import wandb
import time
run_name = f"{args.env_id}__{args.exp_name}__{int(time.time())}"
if args.track:
    import wandb
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        config=vars(args),
        name=run_name,
        save_code=True,
    )

# Create grf env
env = create_env() 

# Set algorithm to train on
trainer = IPPO(env, args)

# Train the policy
trainer.train()

# Log results
if args.track:
    wandb.finish()