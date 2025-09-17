# Utils
from utils import apply_orthogonal_init, log_metrics, record_video, upload_videos_to_wandb, compute_behavioral_diversity, make_pre_step_fill_adv

# Torch
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictSequential
from torchrl.modules import TanhNormal
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

#Logging
from tqdm import tqdm
import time


class PPO:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        mappo = args.mappo == True
        self.global_step = 0
        self.agent_frozen = False
        self.policies = {}
        self.critics = {}
        self.losses = {}
        self.optimizers = {}
        
        
        for group, agents in self.env.group_map.items():
            n_agents = len(agents)
            obs_size = self.env.observation_spec[group, "observation"].shape[-1]
            action_size = self.env.input_spec["full_action_spec"][group, "action"].shape[-1]

            # Policy network
            backbone = torch.nn.Sequential(
                MultiAgentMLP(
                    n_agent_inputs=obs_size,
                    n_agent_outputs=2 * action_size,
                    n_agents=n_agents,
                    centralised=False,
                    share_params=mappo,
                    depth=3,
                    num_cells=[256, 128, 64],
                    device=self.args.device,
                    activation_class=torch.nn.Tanh,
                ),
                NormalParamExtractor(scale_mapping="biased_softplus_1.0"),  # this will just separate the last dimension into two outputs: a loc and a non-negative scale
            )
            apply_orthogonal_init(backbone)
            module = TensorDictModule(
                backbone, 
                in_keys=[(group, "observation")], 
                out_keys=[(group, "loc"), (group, "scale")]
            )

            policy = ProbabilisticActor(
                module=module,
                spec=env.input_spec["full_action_spec"][group]["action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=TanhNormal,
                distribution_kwargs={
                    "low": env.input_spec["full_action_spec"][group]["action"].space.low,
                    "high": env.input_spec["full_action_spec"][group]["action"].space.high,
                },
                return_log_prob=True,
                log_prob_key=(group, "sample_log_prob"),
            )
            self.policies[group] = policy

            # Critic network
            critic_net = MultiAgentMLP(
                n_agent_inputs=obs_size,
                n_agent_outputs=1,
                n_agents=n_agents,
                centralised=mappo,
                share_params=mappo,
                device=args.device,
                depth=3,
                num_cells=[256, 128, 64],
                activation_class=torch.nn.ReLU
            )
            apply_orthogonal_init(critic_net)
            critic = TensorDictModule(
                critic_net, 
                in_keys=[(group, "observation")], 
                out_keys=[(group, "state_value")]
            )
            self.critics[group] = critic

            # Agent loss module
            loss_module = ClipPPOLoss(
                actor_network=policy,
                critic_network=critic,
                clip_epsilon=args.clip_epsilon,
                entropy_coef=args.entropy_eps,
                normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
            )
            loss_module.set_keys(  # We have to tell the loss where to find the keys
                reward=(group, "reward"),
                action=(group, "action"),
                value=(group, "state_value"),
                sample_log_prob=(group, "sample_log_prob"),  
                value_target=(group, "value_target"),  
                # These last 2 keys will be expanded to match the reward shape
                done=(group, "done"),
                terminated=(group, "terminated"),
                advantage=(group, "advantage"),  
            )

            loss_module.make_value_estimator(
                ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda
            )  # Enables GAE inside the PPO loss module
            self.losses[group] = loss_module
            self.optimizers[group] = torch.optim.Adam(loss_module.parameters(), args.learning_rate)

        # Combine all policies into one sequential module
        self.collect_policy = TensorDictSequential(*self.policies.values())
        # Data collector
        self.collector = SyncDataCollector(
            env,
            self.collect_policy,
            device=args.device,
            storing_device=args.device,
            frames_per_batch =args.env_steps_per_batch,
            total_frames=args.total_env_steps,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                args.env_steps_per_batch, device=args.device
            ),  # We store the env_steps_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=args.minibatch_size,  # We will sample minibatches of this size
        )
        
    def train(self):
        pbar = tqdm(total=self.args.n_iters, desc="episode_reward_mean = 0")
        self.global_step = 0
        collector_iter = iter(self.collector)
        next_record_step = self.args.record_steps  
        
        for i in range(self.args.n_iters):
            t0 = time.time()

            collector_start = time.time()
            tensordict_data = next(collector_iter)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            collector_time = time.time() - collector_start  
            tensordict_data = tensordict_data.to(self.args.device)
            steps_in_batch = tensordict_data.batch_size[0]
            self.global_step += steps_in_batch

            # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
            for group in self.env.group_map.keys():
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
                for group in self.env.group_map.keys():
                    self.losses[group].value_estimator(
                        tensordict_data,  # ✅ full TD, not just group slice
                        params=self.losses[group].critic_network_params,
                        target_params=self.losses[group].target_critic_network_params,
                    )
            
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gae_time = time.time() - gae_start
            buffer_start = time.time()
            data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
            self.replay_buffer.extend(data_view)
            buffer_time = time.time() - buffer_start
            
            # Logging diversity metrics
            get_diversity_metrics = compute_behavioral_diversity(tensordict_data)
            diversity_metrics = {f"diversity/{k}": v for k, v in get_diversity_metrics.items()}
            log_metrics(diversity_metrics, step=self.global_step, use_wandb=self.args.track)


            opt_start = time.time()
            for _ in range(self.args.num_epochs):
                for _ in range(self.args.env_steps_per_batch  // self.args.minibatch_size):
                    subdata = self.replay_buffer.sample().to(self.args.device)

                    for group in self.env.group_map.keys():
                        # Check if agent should be frozen
                        if (not self.agent_frozen) and self.global_step >= self.args.agent_training_steps:
                            print(f"Freezing agent at step {self.global_step}")
                            for p in self.losses["agent"].parameters():
                                p.requires_grad = False
                            self.agent_frozen = True

                        # Skip agent PPO update if frozen
                        if group == "agent" and self.agent_frozen:
                            continue                        

                        # Compute loss and update policy
                        loss_vals = self.losses[group](subdata)
                        total_loss = (
                            loss_vals["loss_objective"] +
                            loss_vals["loss_critic"] +
                            loss_vals["loss_entropy"]
                        )                        
                        self.optimizers[group].zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.losses[group].parameters(), self.args.max_grad_norm)
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
            self.collector.update_policy_weights_()
            sync_time = time.time() - sync_start

            total_iteration_time = time.time() - t0

            if self.global_step >= next_record_step:
                print(f"Recording video at global step {self.global_step}")
                record_video(multi_agent_policy=self.collect_policy, device=self.args.device)
                upload_videos_to_wandb(scenario="simple_tag", algorithm=self.args.exp_name, step=self.global_step)
                next_record_step += self.args.record_steps

            # General logging 
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
            for group in self.env.group_map.keys():
                rewards = tensordict_data.get(("next", group, "reward"))
                group_rewards[group] = rewards.mean().item()
                    
            for group, reward_mean in group_rewards.items():
                reward_metrics = {
                    f"charts/episode_reward_{group}_mean": reward_mean,
                }
                log_metrics(reward_metrics, step=self.global_step, use_wandb=self.args.track)

            pbar.set_description(
            "Rewards: " + ", ".join(f"{group}={reward:.2f}" for group, reward in group_rewards.items()),
            refresh=False
            )
            pbar.update()
