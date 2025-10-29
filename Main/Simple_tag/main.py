if __name__ == "__main__":    
    import os
    os.environ["TORCHINDUCTOR_FALLBACK"] = "1"           # Allow fallback to eager mode
    os.environ["TORCHINDUCTOR_DISABLE_CAPTURE"] = "1"       
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    import torch
    from torch import device
    import numpy as np
    from tensordict.nn import set_composite_lp_aggregate
    from config.config import parse_training_args
    from env import create_env
    from gappo import GAPPO
    from HetGPPO import HetGPPO
    from ppo import PPO
    from torchrl.envs import ParallelEnv

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(device)} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
        torch.set_default_device(device)
    else:
        print("CUDA not available, using CPU.")

    # Which algorithm do we want to train
    algorithms = {
        "MAPPO": PPO, # with args.mappo == True
        "IPPO": PPO, # with args.mappo == False
        "GAPPO": GAPPO, # with args.shared_backbone == True
        "IGAPPO": GAPPO, # with args.shared_backbone == False
    }

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
    num_workers = args.number_of_workers
    worker_seeds = [args.seed + i for i in range(num_workers)]
    iter_seeds = iter(worker_seeds)
    env = ParallelEnv(num_workers, lambda idx=None: create_env(seed=worker_seeds[idx] if idx is not None else worker_seeds[0]), share_individual_td=True)

    # Set algorithm to train on
    if args.algorithm not in algorithms:
        raise ValueError(f"Unknown algorithm: {args.algorithm}. Choose from {list(algorithms.keys())}")
    algorithm = algorithms[args.algorithm]
    trainer_algorithm = algorithm(env, args)

    # Train the policy
    trainer_algorithm.train()

    # Log results
    if args.track:
        wandb.finish()