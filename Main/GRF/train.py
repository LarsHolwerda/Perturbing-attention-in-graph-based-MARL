import gc
import torch
from tensordict.nn import TensorDictModule
import sipo
from utils import log_metrics, record_video, upload_videos_to_wandb, compute_behavioral_diversity
import time
from tqdm import tqdm
import time


class Train:
    def __init__(self, args, env, collector, replay_buffer, losses, optimizers):
        self.args = args
        self.env = env
        self.collector = collector
        self.replay_buffer = replay_buffer
        self.losses = losses
        self.optimizers = optimizers
        self.global_step = 0
        self.next_record_step = args.record_steps
        self.perturb_attention_logits = False
        
   
    def train(self):
        # Initialize progress bar
        print("Starting training...")
        pbar = tqdm(total=self.args.n_iters - 1, desc="episode_reward_mean = 0")
        # Initialize collector iterator
        collector_iter = iter(self.collector)
        print("Initialized collector iterator")
        
        # Training loop
        for i in range(self.args.n_iters - 1):
            t0 = time.time()
            collector_start = time.time()
            # Collect data from the collector
            tensordict_data = next(collector_iter)
            if torch.backends.cuda.is_built() and torch.cuda.is_available(): torch.cuda.synchronize()
            collector_time = time.time() - collector_start 
            tensordict_data = tensordict_data.to(self.args.device)
            steps_in_batch = self.args.env_steps_per_batch

            self.global_step += steps_in_batch
            # Update global step in the backbone(s) for attention perturbation scheduling
            if self.args.algorithm in ["GAPPO", "IGAPPO", "PGAPPO", "PIGAPPO"]:
                for policy_name, policy in self.policies.items():
                    policy_module = policy.module  # ModuleList
                    for module in policy_module:  # TensorDictModule
                        if isinstance(module, TensorDictModule):
                            actor_head = module.module
                            if self.args.algorithm in ["GAPPO", "PGAPPO"]:
                                backbone = actor_head.base_model  # Shared backbone
                                backbone.global_step = self.global_step
                            elif self.args.algorithm in ["IGAPPO", "PIGAPPO"]:
                                for backbone in actor_head.base_model:  # Individual backbones
                                    backbone.global_step = self.global_step
                        else:
                            continue
            
            # add intrinsic reward to push diversity
            sipo_time = time.time()
            tensordict_data = self.sipo.compute_intrinsic_reward(tensordict_data, self.global_step)

            # Store sampled states in the archive trajectories
            self.sipo.store_archive(self.args, i, tensordict_data)
            sipo_time = time.time() - sipo_time


            # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
            tensordict_data.set(
                ("next", "player", "done"),
                tensordict_data.get(("next", "done"))
                .unsqueeze(-1)
                .expand(tensordict_data.get_item_shape(("next", self.env.reward_key))),
            )
            tensordict_data.set(
                ("next", "player", "terminated"),
                tensordict_data.get(("next", "terminated"))
                .unsqueeze(-1)
                .expand(tensordict_data.get_item_shape(("next", self.env.reward_key))),
            )
            

            gae_start = time.time()
            # Compute GAE and add it to the data
            with torch.no_grad():
                self.loss_module.value_estimator(
                    tensordict_data,
                    params=self.loss_module.critic_network_params,
                    target_params=self.loss_module.target_critic_network_params,
                )
            if torch.backends.cuda.is_built() and torch.cuda.is_available(): torch.cuda.synchronize()
            gae_time = time.time() - gae_start
            buffer_start = time.time()
            # Flatten the batch size to shuffle data
            data_view = tensordict_data.reshape(-1) 
            # Add the flattened data to the replay buffer
            self.replay_buffer.extend(data_view)

            buffer_time = time.time() - buffer_start

            # Logging diversity metrics
            get_diversity_metrics = compute_behavioral_diversity(tensordict_data, self.policy, self.args.n_agents)
            diversity_metrics = {f"diversity/{k}": v for k, v in get_diversity_metrics.items()}
            log_metrics(diversity_metrics, step=self.global_step, use_wandb=self.args.track)
            print("before_optimization")
            opt_start = time.time()

            for epoch in range(self.args.num_epochs):  # Loop over epochs
                for _ in range(self.args.env_steps_per_batch  // self.args.minibatch_size):  # Loop over mini-batches
                    subdata = self.replay_buffer.sample() # Sample a mini-batch from the replay buffer
                    subdata = subdata.to(self.args.device)
                    # Compute loss
                    loss_vals = self.losses(subdata)
                    total_loss = (
                        loss_vals["loss_objective"] +
                        loss_vals["loss_critic"] +
                        loss_vals["loss_entropy"]
                    )

                    # Clear gradients
                    self.optimizers.zero_grad()
                    # Backpropagate the loss
                    total_loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.losses.parameters(), self.args.max_grad_norm)
                    # Update parameters
                    self.optimizers.step()

                    # Logging metrics for adversary
                    log_metrics({
                        f"loss/loss_pg": loss_vals["loss_objective"].detach().to("cpu", non_blocking=True),
                        f"loss/loss_v": loss_vals["loss_critic"].detach().to("cpu", non_blocking=True),
                        f"loss/loss_entropy": loss_vals["loss_entropy"].detach().to("cpu", non_blocking=True),
                        f"loss/entropy": loss_vals["entropy"].detach().to("cpu", non_blocking=True),
                        f"loss/approx_kl": loss_vals["kl_approx"].detach().to("cpu", non_blocking=True),
                        f"loss/clip_fraction": loss_vals["clip_fraction"].detach().to("cpu", non_blocking=True),
                    }, self.global_step, self.args.track)

            if torch.backends.cuda.is_built() and torch.cuda.is_available(): torch.cuda.synchronize()
            opt_time = time.time() - opt_start

            sync_start = time.time()
            # Synchronize policy weights into the collector's internal policy
            self.collector.update_policy_weights_()
            sync_time = time.time() - sync_start

            total_iteration_time = time.time() - t0
            print(f"Iteration {i+1}/{self.args.n_iters - 1} took {total_iteration_time:.3f}s (collector: {collector_time:.3f}s, SIPO: {sipo_time:.3f}s, GAE: {gae_time:.3f}s, buffer: {buffer_time:.3f}s, optimization: {opt_time:.3f}s, sync: {sync_time:.3f}s)")
            # Record videos and upload to wandb
            if self.global_step >= self.next_record_step:
                print(f"Recording video at global step {self.global_step}")
                record_video(self.env, self.args.exp_name, self.policy, self.args.n_agents, self.args.device, num_episodes=self.args.num_episodes_to_record)
                upload_videos_to_wandb(scenario=self.args.env_id, algorithm=self.args.exp_name, step=self.global_step)
                self.next_record_step += self.args.record_steps

            # Time and reward logging per iteration to wandb
            time_metrics = {
                "timing/total_iteration_time": total_iteration_time,
                "timing/collector_time": collector_time,
                "timing/sipo_time": sipo_time,
                "timing/gae_time": gae_time,
                "timing/buffer_time": buffer_time,
                "timing/optimization_time": opt_time,
                "timing/sync_time": sync_time,
            }
            log_metrics(time_metrics, step=self.global_step, use_wandb=self.args.track)

            # Reward logging
            done = tensordict_data.get(("next", "player", "done"))
            episode_reward_mean = (
                tensordict_data.get(("next", "player", "episode_reward"))[done].mean().item()
            )
            log_metrics({"rewards/episode_reward_players": episode_reward_mean}, step=self.global_step, use_wandb=self.args.track)

            # Update progress bar
            pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
            pbar.update()
            print("done iteration")
        
        del self.collector