if __name__ == "__main__":    
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    import torch
    import numpy as np
    from tensordict.nn import set_composite_lp_aggregate
    from config import parse_training_args
    from env import create_env
    from gappo import GAPPO
    from HetGPPO import HetGPPO
    from ppo import PPO
    from torchrl.envs import ParallelEnv

    # disable log-prob aggregation
    set_composite_lp_aggregate(False).set()

    # Get command line arguments
    args = parse_training_args()

    # Set seed
    print(f"Starting run with seed {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

    # Log to wandb
    import wandb
    import time
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            save_code=True,
        )

    # Create env
    num_workers = 2
    worker_seeds = [args.seed + i for i in range(num_workers)]
    iter_seeds = iter(worker_seeds)
    env = ParallelEnv(num_workers, lambda: create_env(seed=next(iter_seeds)), share_individual_td=True)

    # Set algorithm to train on
    trainer_algorithm = PPO(env, args)

    # Train the policy
    trainer_algorithm.train()

    # Log results
    if args.track:
        wandb.finish()