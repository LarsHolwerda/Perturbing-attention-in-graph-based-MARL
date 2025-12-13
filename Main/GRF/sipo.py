import torch
import copy
import numpy as np
from utils import log_metrics
from w_discriminator import WassersteinDiscriminator



class SIPO_WD:
    def __init__(self, args):
        self.args = args
        self.alpha = args.alpha
        self.delta = args.delta
        self.lambda_lr = args.lambda_lr
        self.wd_lr = args.wd_lr
        self.lambda_max = args.lambda_max
        self.opt_eps = args.opt_eps
        
        # List of tensors of each iteration, each tensor is [num_states, obs_dim]
        self.archive = [] 
        self.collected_trajectories = []
        # Counter to select which trajectories will be stored
        self.archive_counter = 0

        # Current iterations WD critic and optimizer
        self.current_wd = None
        self.current_wd_opt = None

        # Previous iterations critic and lagrange multipliers
        self.lambdas = []
        self.wd_critics = []
    
    

    def start_new_iteration(self):
        # Reset lagrange multipliers for all previous archive entries
        self.lambdas = [torch.zeros(1, device=self.args.device) for _ in self.archive]

        # Create a new WD critic for this iteration
        self.current_wd = WassersteinDiscriminator(
            type_="frame_stack",
            obs_dim=self.args.obs_dim,
            act_dim=None,  
            act_space=None,
            hidden_dim=64
        ).to(self.args.device)
        # Create optimizer for the current WD critic
        self.current_wd_opt = torch.optim.RMSprop(self.current_wd.parameters(), lr=self.wd_lr, eps=self.opt_eps)

        # Each policy iteration has their own archive list
        self.archive.append([])  

    # Compute intrinsic reward, update WD critic, and update lagrange multipliers
    def compute_intrinsic_reward(self, tensordict_data, global_step):
        # Get observations from current batch and reshape when necessary
        obs = tensordict_data.get(("player", "observation"))
        E, B, N, O = obs.shape
        # Update Wasserstein critic using current batch + archived states
        
        
        iter_idx = len(self.archive) - 1 # Use the most recent archive for critic update
        if len(self.archive[iter_idx]) > 0:
            # Sample a block from the archive
            idx = np.random.randint(0, len(self.archive[iter_idx]))
            archived_obs = self.archive[iter_idx][idx].to(self.args.device)
            
            current_batch_scores = self.current_wd(obs, None, None, None) # output of critic for current states
            archived_scores = self.current_wd(archived_obs, None, None, None) # output of critic for archived states

            wd_loss = -(current_batch_scores.mean() - archived_scores.mean()) # Wasserstein loss

            # optimize critic
            self.current_wd_opt.zero_grad() # Reset gradients   
            wd_loss.backward()
            self.current_wd_opt.step()
        else:
            wd_loss = torch.tensor(0.0, device=self.args.device)

        # Compute intrinsic reward from all previous critics
        int_r_total = torch.zeros((E, B, N), device=self.args.device)

        # For each previous critic, compute intrinsic reward
        with torch.no_grad():
            for j, (lambda_j, critic_j, archive_j) in enumerate(zip(self.lambdas, self.wd_critics, self.archive)):             
                # Sample a block of length B from the previous archive
                idx = np.random.randint(0, len(archive_j))
                previous_archived_obs = archive_j[idx].to(self.args.device)   
                print("archive_j[idx].shape =", archive_j[idx].shape)
                # Compute critic scores for the current batch and archive mean score
                critic_j = critic_j.to(self.args.device)
                critic_scores = critic_j(obs, None, None, None)
                print("critic_scores.mean() =", critic_scores.mean())
                archive_mean = critic_j(previous_archived_obs, None, None, None).mean()
                print("archive_mean =", archive_mean)
                # Compute intrinsic reward per step according to the wasserstein critic
                r_j = critic_scores - archive_mean
                r_j = r_j.squeeze(-1)
                # Compute (lagrange multiplier * intrinsic reward) for the actually added intrinsic reward
                int_r_total += lambda_j * r_j

                # Gradient ascent on λ_j
                R_j_int = r_j.sum() 
                new_lambda = self.lambdas[j] + self.lambda_lr * (-R_j_int + self.delta)
                self.lambdas[j] = new_lambda.clamp(0.0, self.lambda_max)

                # Logging
                r_j_mean = r_j.mean().item()
                lambda_update = (-r_j_mean + self.delta)
                lambda_value = self.lambdas[j].item()

                log_metrics({
                    f"sipo/critic_{j}/total_intrinsic_reward": R_j_int,
                    f"sipo/critic_{j}/lambda_update": lambda_update,
                    f"sipo/critic_{j}/lambda_value": lambda_value,
                    f"sipo/critic_{j}/archive_mean": archive_mean,
                }, step=global_step, use_wandb=self.args.track)
                

        int_r = int_r_total.unsqueeze(-1).detach()  # [num_envs, B, N, 1] 

        # Logging overall metrics
        log_metrics({
            "sipo/intrinsic_reward_mean": int_r.mean().item(),
            "sipo/scaled_intrinsic_reward_mean": (self.alpha * int_r_total).mean().item(),
            "sipo/wd_loss": wd_loss.item() if len(self.archive) > 0 else 0.0,
        }, step=global_step, use_wandb=self.args.track)

        # Add intrinsic reward to environment reward
        env_reward = tensordict_data.get(("next", "player", "reward"))  
        new_reward = env_reward + self.alpha * int_r.detach()
        tensordict_data.set(("next", "player", "reward"), new_reward)
        
        # Clean up to free memory
        del obs
        torch.cuda.empty_cache()
        return tensordict_data

    # Store sampled states from current batch into collected_trajectories
    def store_archive(self, args, cur_iteration, tensordict_data):
        # Decide which environments to store trajectories from
        if args.n_iters - cur_iteration < 30:
            obs = tensordict_data.get(("player", "observation")).detach().cpu().clone()
            self.archive[-1].append(obs)

    # Save critic from the current iteration
    def save_critic(self):
        # Save the critic of this iteration
        if self.current_wd is not None:
            self.wd_critics.append(copy.deepcopy(self.current_wd).cpu().eval())
