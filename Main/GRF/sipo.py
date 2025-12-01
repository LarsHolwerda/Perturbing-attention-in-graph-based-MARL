import torch
import copy

class WD_Critic(torch.nn.Module):
    def __init__(self, obs_dim, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )
    def forward(self, obs):
        return self.net(obs)


class SIPO_WD:
    def __init__(self, args):
        self.args = args
        self.alpha = args.alpha
        self.delta = args.delta
        self.lambda_lr = args.lambda_lr
        self.wd_lr = args.wd_lr
        self.lambda_max = args.lambda_max
        
        # list of tensors of each iteration, each tensor is [num_states, obs_dim]
        self.archive = [] 
        self.collected_states = []
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
        self.current_wd = WD_Critic(self.args.obs_dim).to(self.args.device)
        self.current_wd_opt = torch.optim.Adam(self.current_wd.parameters(), lr=self.wd_lr)

    # Compute intrinsic reward, update WD critic, and update lagrange multipliers
    def compute_intrinsic_reward(self, tensordict_data):
        # Get observations from current batch and reshape when necessary
        obs = tensordict_data.get(("player", "observation"))
        print(f"obs shape: {obs.shape}")
        if obs.ndim == 4:          
            E, B, N, O = obs.shape
            flat_obs = obs.reshape(E * B * N, O) # [num_envs * B * N, obs_dim]
        elif obs.ndim == 3:        
            B, N, O = obs.shape
            flat_obs = obs.reshape(B * N, O)

        # Update Wasserstein critic using current batch + archived states
        if len(self.archive) > 0:
            # Concatenate all past archive states
            all_archive_states = torch.cat(self.archive, dim=0)

            # Sample batch same size as current data
            idx = torch.randint(
                low=0,
                high=all_archive_states.size(0),
                size=(flat_obs.size(0),),
                device=self.args.device,
            ) # Get random indices for sampling
            archive_batch = all_archive_states[idx].to(self.args.device) # Sampled archived states

            current_batch_scores = self.current_wd(flat_obs) # output of critic for current states
            archived_scores = self.current_wd(archive_batch) # output of critic for archived states

            wd_loss = -(current_batch_scores.mean() - archived_scores.mean()) # Wasserstein loss

            # optimize critic
            self.current_wd_opt.zero_grad() # Reset gradients   
            wd_loss.backward()
            self.current_wd_opt.step()

        # Compute intrinsic reward from all previous critics
        r_int_total = torch.zeros((E, B, N), device=self.args.device)

        # For each previous critic, compute intrinsic reward
        for j, (lambda_j, critic_j, archive_j) in enumerate(zip(self.lambdas, self.wd_critics, self.archive)):
            # Compute critic scores for the current batch and archive mean score
            critic_scores = critic_j(flat_obs).reshape(E, B, N)
            archive_mean = critic_j(archive_j.to(self.args.device)).mean()

            # Compute intrinsic reward per step according to the wasserstein critic
            r_j = critic_scores - archive_mean

            # Compute (lagrange multiplier * intrinsic reward) for the actually added intrinsic reward
            r_int_total += lambda_j * r_j

        r_int = r_int_total.unsqueeze(-1).detach()  # [num_envs, B, N, 1] 

        # Add intrinsic reward to environment reward
        env_reward = tensordict_data.get(("next", "player", "reward"))  
        new_reward = env_reward + self.alpha * r_int
        tensordict_data.set(("next", "player", "reward"), new_reward)


        # Update langrange multipliers for previous critics
        if len(self.archive) > 0:
            r_j_list = []
            for j, (critic_j, archive_j) in enumerate(zip(self.wd_critics, self.archive)):
                critic_scores = critic_j(flat_obs).reshape(E, B, N)
                archive_mean = critic_j(archive_j.to(self.args.device)).mean()
                r_j = critic_scores - archive_mean
                r_j_list.append(r_j.mean())

            # Gradient ascent on λ_j
            for j in range(len(self.lambdas)):
                new_lambda = self.lambdas[j] + self.lambda_lr * (-r_j_list[j] + self.delta)
                self.lambdas[j] = new_lambda.clamp(0.0, self.lambda_max)

        return tensordict_data

    def store_archive(self, tensordict_data):
        obs = tensordict_data.get(("player", "observation")).detach().cpu()
        if obs.ndim == 4:       # [T, B, N, D]
            E, B, N, O = obs.shape
            obs = obs.reshape(E * B * N, O)
        elif obs.ndim == 3:     # [B, N, D]
            B, N, O = obs.shape
            obs = obs.reshape(B * N, O)
        
        num_states = obs.size(0)
        k = max(1, int(0.10 * num_states))
        idx = torch.randperm(num_states, device=obs.device)[:k]
        sampled_states = obs[idx]            
        self.collected_states.append(sampled_states)

    # Save trajectories from the current iteration
    def save_archive(self, collected_states):
        # collected_trajectories: [T, obs_dim]
        with torch.no_grad():
            traj = collected_states.cpu()
            self.archive.append(traj)

            # Save the critic of this iteration
            self.wd_critics.append(copy.deepcopy(self.current_wd).eval())
