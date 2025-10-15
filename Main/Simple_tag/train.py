import torch
from utils import coo_to_dense_weights, coo_to_dense_weights_batched, log_metrics, record_video, upload_videos_to_wandb, compute_behavioral_diversity
import time
from tqdm import tqdm
import time, sys, psutil, os


class Train:
    def __init__(self, args, env, collector, replay_buffer, losses, optimizers, group_map):
        self.args = args
        self.env = env
        self.collector = collector
        self.replay_buffer = replay_buffer
        self.losses = losses
        self.optimizers = optimizers
        self.group_map = group_map  
        self.global_step = 0
        self.agent_frozen = False
        if self.args.algorithm in ["GAPPO", "IGAPPO"]:
            self.analysis_data = {
                "observations": [],
                "adjacency": [],
                "actions": [],
                "done": [],
            }

    def train(self):
        # Initialize progress bar
        pbar = tqdm(total=self.args.n_iters, desc="episode_reward_mean = 0")
        self.global_step = 0
        next_record_step = self.args.record_steps  
        # Initialize collector iterator
        collector_iter = iter(self.collector)

        # Training loop
        for i in range(self.args.n_iters):
            t0 = time.time()
            collector_start = time.time()
            # Collect data from the collector
            tensordict_data = next(collector_iter)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            collector_time = time.time() - collector_start 
            print(f"[DEBUG] Collector time: {collector_time:.3f}s")
            tensordict_data = tensordict_data.to(self.args.device)
            steps_in_batch = self.args.env_steps_per_batch
            self.global_step += steps_in_batch

            # For GAPPO/IGAPPO we want to store the states, actions and adjacency weights for analysis
            if self.args.algorithm in ["GAPPO", "IGAPPO"] and self.global_step > self.args.total_env_steps - self.args.env_steps_to_analyze:
                for group in self.group_map.keys():
                    obs = tensordict_data.get((group, "observation")).detach().cpu()
                    acts = tensordict_data.get((group, "action")).detach().cpu()
                    terminated = tensordict_data.get(("next", group, "terminated")).detach().cpu()
                    print(f"[DEBUG] analysis data collection, iter={i}, global_step={self.global_step}, obs shape={obs.shape}, acts shape={acts.shape}, done shape={terminated.shape}")
                    with torch.no_grad():
                        if group == "adversary":
                            policy_module = self.policies[group].module[0]
                            actor_head = policy_module.module
                            if self.args.algorithm == "GAPPO":
                                base_model = actor_head.base_model    
                                _ = base_model(obs)
                                
                                print(f"[DEBUG] before adj conversion, iter={i}, global_step={self.global_step}")
                                sys.stdout.flush()
                                t_debug0 = time.time()
                                edge_index, att_values = base_model.gat.last_att_weights
                                geo_batch = base_model.geo_batch
                                t_start_conv = time.time()
                                # Convert attention to dense adjacency per sample
                                adj_weights = coo_to_dense_weights_batched(
                                    edge_index, att_values, geo_batch.batch, self.args.n_adversaries, self.args.number_of_workers
                                )
                                t_end_conv = time.time()

                                print(f"[DEBUG] after adj conversion, took={t_end_conv - t_start_conv:.3f}s, total since debug start={t_end_conv - t_debug0:.3f}s")
                                print("Memory % used:", psutil.virtual_memory().percent)
                                sys.stdout.flush()                
                            
                            elif self.args.algorithm == "IGAPPO":
                                adj_weights = []
                                for net in policy_module.base_model:
                                    edge_index, att_values = net.gat.last_att_weights
                                    att_dense = coo_to_dense_weights(edge_index, att_values, self.args.n_adversaries)[0].detach().cpu()
                                    adj_weights.append(att_dense)
                                adj_weights = torch.stack(adj_weights)

                    # Append to analysis data
                    analysis_time = time.time()
                    self.analysis_data["observations"].append(obs)
                    self.analysis_data["actions"].append(acts)
                    if group == "adversary":
                        self.analysis_data["adjacency"].append(adj_weights)
                    self.analysis_data["done"].append(terminated)
                    analysis_time_final = time.time() - analysis_time
                    print(f"[DEBUG] analysis data append took {analysis_time_final:.3f}s")
            # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
            for group in self.group_map.keys():
                group_shape = tensordict_data.get_item_shape(("next", group, "reward"))
                tensordict_data.set(
                    ("next", group, "done"),
                    tensordict_data.get(("next", "done")).unsqueeze(-1).expand(group_shape),
                )
                tensordict_data.set(
                    ("next", group, "terminated"),
                    tensordict_data.get(("next", "terminated")).unsqueeze(-1).expand(group_shape),
                )

            gae_start = time.time()
            # Compute GAE and add it to the data
            with torch.no_grad():
                for group in self.group_map.keys():
                    self.losses[group].value_estimator(
                        tensordict_data,
                        params=self.losses[group].critic_network_params,
                        target_params=self.losses[group].target_critic_network_params,
                    )
        
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gae_time = time.time() - gae_start
            buffer_start = time.time()

            # Flatten the batch size to shuffle data
            data_view = tensordict_data.reshape(-1) 
            # Add the flattened data to the replay buffer
            self.replay_buffer.extend(data_view)

            buffer_time = time.time() - buffer_start
            
            # Logging diversity metrics
            get_diversity_metrics = compute_behavioral_diversity(self.collect_policy, tensordict_data, self.args.n_adversaries)
            diversity_metrics = {f"diversity/{k}": v for k, v in get_diversity_metrics.items()}
            log_metrics(diversity_metrics, step=self.global_step, use_wandb=self.args.track)


            opt_start = time.time()
            # Loop over epochs
            for _ in range(self.args.num_epochs):
                # Loop over mini-batches
                for _ in range(self.args.env_steps_per_batch  // self.args.minibatch_size):
                    # Sample a mini-batch from the replay buffer
                    subdata = self.replay_buffer.sample()

                    for group in self.group_map.keys():
                        # Check if agent should be frozen
                        if (not self.agent_frozen) and self.global_step >= self.args.agent_training_steps:
                            print(f"Freezing agent at step {self.global_step}")
                            for p in self.losses["agent"].parameters():
                                p.requires_grad = False
                            self.agent_frozen = True

                        # Skip agent PPO update if frozen
                        if group == "agent" and self.agent_frozen:
                            continue                        

                        # Compute loss
                        loss_vals = self.losses[group](subdata)
                        total_loss = (
                            loss_vals["loss_objective"] +
                            loss_vals["loss_critic"] +
                            loss_vals["loss_entropy"]
                        )
                        # Clear gradients
                        self.optimizers[group].zero_grad()
                        # Backpropagate the loss
                        total_loss.backward()
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(self.losses[group].parameters(), self.args.max_grad_norm)
                        # Update parameters
                        self.optimizers[group].step()
                            

                        # Logging metrics for adversary
                        log_metrics({
                            f"{group}/loss_pg": loss_vals["loss_objective"].item(),
                            f"{group}/loss_v": loss_vals["loss_critic"].item(),
                            f"{group}/loss_entropy": loss_vals["loss_entropy"].item(),
                            f"{group}/entropy": loss_vals["entropy"].item(),
                            f"{group}/approx_kl": loss_vals["kl_approx"].item(),
                            f"{group}/clip_fraction": loss_vals["clip_fraction"].item(),
                        }, step=self.global_step, use_wandb=self.args.track)
        

            if torch.cuda.is_available(): torch.cuda.synchronize()
            opt_time = time.time() - opt_start

            sync_start = time.time()
            # Synchronize policy weights into the collector's internal policy
            self.collector.update_policy_weights_()
            sync_time = time.time() - sync_start

            total_iteration_time = time.time() - t0

            # Record videos and upload to wandb
            if self.global_step >= next_record_step:
                print(f"Recording video at global step {self.global_step}")
                record_video(multi_agent_policy=self.collect_policy, algorithm=self.args.exp_name, n_agents=self.args.n_agents, n_adv=self.args.n_adversaries, device=self.args.device)
                upload_videos_to_wandb(scenario="simple_tag", algorithm=self.args.exp_name, step=self.global_step)
                next_record_step += self.args.record_steps

            # Time and reward logging per iteration to wandb
            time_metrics = {
                "timing/total_iteration_time": total_iteration_time,
                "timing/collector_time": collector_time,
                "timing/gae_time": gae_time,
                "timing/buffer_time": buffer_time,
                "timing/optimization_time": opt_time,
                "timing/sync_time": sync_time,
            }
            log_metrics(time_metrics, step=self.global_step, use_wandb=self.args.track)

            group_rewards = {}
            for group in self.group_map.keys():
                rewards = tensordict_data.get(("next", group, "reward"))
                group_rewards[group] = rewards.mean().item()
                    
            for group, reward_mean in group_rewards.items():
                reward_metrics = {
                    f"charts/episode_reward_{group}_mean": reward_mean,
                }
                log_metrics(reward_metrics, step=self.global_step, use_wandb=self.args.track)

            # Update progress bar
            pbar.set_description(
            "Rewards: " + ", ".join(f"{group}={reward:.2f}" for group, reward in group_rewards.items()),
            refresh=False
            )
            pbar.update()

        output_file = f"{self.args.env_id}__{self.args.exp_name}__{self.args.seed}__{int(time.time())}.pt"
        torch.save(self.analysis_data, f"analysis/{output_file}")
        print(f"Analysis data saved to analysis/{output_file}")
