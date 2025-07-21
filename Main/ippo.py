# Utils
from utils import apply_orthogonal_init, log_metrics, record_video, upload_videos_to_wandb

# Torch
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical

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





class IPPO:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.device = args.device
        self.global_step = 0

        policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["player", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=19,  # n_actions_per_agents
        n_agents=args.n_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=False,
        device=args.device,
        depth=3,
        num_cells=[256, 128, 64],
        activation_class=torch.nn.ReLU,
        )
        apply_orthogonal_init(policy_net)


        policy_module = TensorDictModule(
            policy_net,
            in_keys=[("player", "observation")],
            out_keys=[("player", "logits")],
        )

        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec_unbatched,
            in_keys=[("player", "logits")],
            out_keys=[env.action_key],
            distribution_class=Categorical,
            return_log_prob=True,
        )  # we'll need the log-prob for the PPO loss


        critic_net = MultiAgentMLP(
            n_agent_inputs=env.observation_spec["player", "observation"].shape[-1],
            n_agent_outputs=1,  # 1 value per agent
            n_agents=args.n_agents,
            centralised=False,
            share_params=False,
            device=args.device,
            depth=3,
            num_cells=[256, 128, 64],
            activation_class=torch.nn.ReLU,
        )
        apply_orthogonal_init(critic_net)

        self.critic = TensorDictModule(
            module=critic_net,
            in_keys=[("player", "observation")],
            out_keys=[("player", "state_value")],
        )


        self.collector = SyncDataCollector(
            env,
            self.policy,
            device=args.device,
            storing_device=args.device,
            frames_per_batch =args.env_steps_per_batch,
            total_frames=args.total_env_steps
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                args.env_steps_per_batch, device=args.device
            ),  # We store the env_steps_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=args.minibatch_size,  # We will sample minibatches of this size
        )

        self.loss_module = ClipPPOLoss(
            actor_network=self.policy,
            critic_network=self.critic,
            clip_epsilon=args.clip_epsilon,
            entropy_coef=args.entropy_eps,
            normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
        )
        self.loss_module.set_keys(  # We have to tell the loss where to find the keys
            reward=env.reward_key,
            action=env.action_key,
            value=("player", "state_value"),
            # These last 2 keys will be expanded to match the reward shape
            done=("player", "done"),
            terminated=("player", "terminated"),
        )


        self.loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda
        )  # Enables GAE inside the PPO loss module
        self.GAE = self.loss_module.value_estimator

        self.optim = torch.optim.Adam(self.loss_module.parameters(), args.learning_rate)

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
            # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
            gae_start = time.time()
            with torch.no_grad():
                self.GAE(
                    tensordict_data,
                    params=self.loss_module.critic_network_params,
                    target_params=self.loss_module.target_critic_network_params,
                )  # Compute GAE and add it to the data
            if torch.cuda.is_available(): torch.cuda.synchronize()
            gae_time = time.time() - gae_start
            buffer_start = time.time()
            data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
            self.replay_buffer.extend(data_view)
            buffer_time = time.time() - buffer_start

            opt_start = time.time()
            for _ in range(self.args.num_epochs):
                for _ in range(self.args.env_steps_per_batch  // self.args.minibatch_size):
                    subdata = self.replay_buffer.sample()
                    subdata = subdata.to(self.args.device)
                    loss_vals = self.loss_module(subdata)

                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    loss_value.backward()

                    torch.nn.utils.clip_grad_norm_(
                        self.loss_module.parameters(), self.args.max_grad_norm
                    ) 

                    self.optim.step()
                    self.optim.zero_grad()

                    # Logging metrics
                    pg_loss = loss_vals["loss_objective"]
                    v_loss = loss_vals["loss_critic"]
                    entropy_loss = loss_vals["loss_entropy"]
                    entropy = loss_vals["entropy"]
                    approx_kl = loss_vals["kl_approx"]
                    clipfrac = loss_vals["clip_fraction"]

                    inner_metrics = {
                        "charts/learning_rate": self.optim.param_groups[0]["lr"],
                        "losses/loss_pg": pg_loss.item(),
                        "losses/loss_v": v_loss.item(),
                        "losses/loss_entropy": entropy_loss.item(),
                        "losses/entropy": entropy.item(),
                        "losses/approx_kl": approx_kl.item(),
                        "charts/clip_fraction": clipfrac.item(),
                    }
                    log_metrics(inner_metrics, step=self.global_step, use_wandb=self.args.track)

            if torch.cuda.is_available(): torch.cuda.synchronize()
            opt_time = time.time() - opt_start

            sync_start = time.time()
            self.collector.update_policy_weights_()
            sync_time = time.time() - sync_start

            total_iteration_time = time.time() - t0

            if self.global_step >= next_record_step:
                print(f"Recording video at global step {self.global_step}")
                record_video(self.env, self.policy, self.device, num_episodes=1)
                upload_videos_to_wandb(scenario="academy_3_vs_1_with_keeper", algorithm="ippo", step=self.global_step)
                next_record_step += self.args.record_steps

            # Logging
            done = tensordict_data.get(("next", "player", "done"))
            episode_reward_mean = (
                tensordict_data.get(("next", "player", "episode_reward"))[done].mean().item()
            )
            outer_metrics = {
                "charts/episode_reward_mean": episode_reward_mean,
                "timing/gae_time": gae_time,
                "timing/replay_buffer_time": buffer_time,
                "timing/optimization_time": opt_time,
                "timing/sync_time": sync_time,
                "timing/collector_time": collector_time,
                "timing/total_iter_time": total_iteration_time,
            }
            log_metrics(outer_metrics, step=self.global_step, use_wandb=self.args.track)
            pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
            pbar.update()

        # Save the trained policy
        torch.save(self.policy.state_dict(), "trained_policies/ippo_policy.pt")
         