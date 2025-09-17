# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd
from torch import TensorType

from torch_geometric.data import Batch
from torch_geometric.data.data import Data
from torch_geometric import utils as geo_utils
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.utils import unbatch, unbatch_edge_index, to_edge_index

# Utils
from utils import apply_orthogonal_init, log_metrics, record_video, upload_videos_to_wandb, compute_behavioral_diversity

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


def create_geo_batch(m_obs, m_adj):
    """
      m_obs: [batch_size, n_agents, embedding_dim] batch of observations of agents
      m_adj: [batch_size, n_agents, n_agents] batch of adjacency matrices
    """
    # extract batch_size from m_obs:
    batch_size = m_obs.shape[0]
    # list of Data objects that will be converted into a Pytorch Geometric batch
    data_lst = []
    for i in range(batch_size):
        # extract observation of sample
        sample_obs = m_obs[i]
        # extract adjacency of sample
        sample_adj = m_adj[i]
        # convert adjacency matrix to sparse COO format
        sample_adj, _ = geo_utils.dense_to_sparse(sample_adj)
        # create Pytorch Geometric data object
        d = Data(x=sample_obs, edge_index=sample_adj)
        data_lst.append(d)
    # create a batch from the list of Data objects
    batch = Batch.from_data_list(data_lst)
    return batch


class MLPEncoder(nn.Module):
    """
        Encoder: creates embedding of node observations
    """
    def __init__(self, input_features, embedding_dim=128):
        super(MLPEncoder, self).__init__()
        self.layer1 = nn.Linear(input_features, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, embedding_dim)
        apply_orthogonal_init(self)

    def forward(self, x):
        """
            x: Tensor([batch_size, observation_shape]) observation of an agent

            returns: x: Tensor([batch_size, embedding_dim]) embedded observations of agent
        """
        # encode the received observation with MLP
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # return embedded observation of agent
        return x


class GATLayer(nn.Module):
    """
        Graph Attention convolution mechanism: Creates latent features by combining agent's observations with
        observations of neighbors
    """
    def __init__(self, embedding_dim=128, heads=8):
        super(GATLayer, self).__init__()
        # Initialize GAT-layer
        self.gat_layer = GATv2Conv(embedding_dim, embedding_dim, heads=heads, concat=False)

    def forward(self, x, edge_index):
        """
            x: Tensor([n_agents, embedding_dim]): each agent is seen as a node in the graph with it's embedding as the
                node feature
            edge_index: Tensor(): the topology of the 'x' graph. The nodes of communicating agents are connected with
                each other

            returns: latent_features, att_weights: computed latent features and attention weights
        """
        # create latent features and attention weights
        latent_features, att_weights = self.gat_layer(x, edge_index, return_attention_weights=True)
        return latent_features, att_weights


class ActionLayer(nn.Module):
    """
        Action Layer: computes actions based on created latent features
    """
    def __init__(self, action_size, embedding_dim=128):
        super(ActionLayer, self).__init__()
        # linear layer computes logits based on the latent features
        self.fc1 = nn.Linear(embedding_dim*2, 128)
        self.fc2 = nn.Linear(128, 2 * action_size)
        apply_orthogonal_init(self)

    def forward(self, i1, i2):
        x = torch.cat([i1, i2], dim=-1)
        x = F.relu(self.fc1(x))
        # compute logits based on latent features
        action_parameters = self.fc2(x)
        
        # return computed logits
        return action_parameters


class Init_GAPPO(nn.Module):
    def __init__(self, action_size, n_agents, input_features, hidden_dim=128, save_weights=False):
        super().__init__()

        self.n_agents = n_agents
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.action_size = action_size

        # Layers
        self.encoder = MLPEncoder(input_features=self.input_features, embedding_dim=self.hidden_dim)
        self.gat = GATLayer(embedding_dim=self.hidden_dim, heads=8)
        self.action_layer = ActionLayer(action_size=action_size, embedding_dim=self.hidden_dim)
        self.value_proc = lambda i1, i2: torch.cat([i1, i2], dim=-1)
        self.value_branch = nn.Sequential(
            nn.Linear(self.hidden_dim*2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        apply_orthogonal_init(self.value_branch)


    def forward(self, global_obs, adj):
        """
        Inputs:
            global_obs: Tensor [B, N, obs_dim]
            adj:        Tensor [B, N, N]

        Returns:
            logits:     Tensor [B, N, num_actions]
            values:     Tensor [B, N, 1]
        """
        if global_obs.dim() == 2:
            global_obs = global_obs.unsqueeze(0)
        B, N, obs_dim = global_obs.shape
        assert N == self.n_agents

        # Create adjacency matrix, its a fully connected graph
        adj = torch.ones(B, N, N, device=global_obs.device)
        

        # 1. Encode observations using MLP encoder
        encoded = self.encoder(global_obs)

        # 2. Enable agent communication with GAT-layer
        geo_batch = create_geo_batch(encoded, adj)
        rel, _ = self.gat(geo_batch.x, geo_batch.edge_index)
        rel_unbatched = torch.stack(unbatch(rel, geo_batch.batch))  # [B, N, hidden_dim]

        # 3. Compute action probabilities based on the latent features
        action_parameters = self.action_layer(encoded, rel_unbatched)  # [B, N, num_actions]
        loc, scale = NormalParamExtractor()(action_parameters) # [B * N, num_actions]
        values = self.value_branch(self.value_proc(encoded, rel_unbatched))  # [B, N, 1]
        values = values.reshape(-1, 1) # [B * N, 1]
    
        return loc, scale, values

class ActorHead(nn.Module):
    def __init__(self, base_model, n_agents):
        super().__init__()
        self.base_model = base_model
        self.n_agents = n_agents

    def forward(self, obs, adj):
        loc, scale, _ = self.base_model(obs, adj)
        batch_shape = obs.shape[:-1]  # [B, N] or [N]
        loc = loc.view(*batch_shape, -1)
        scale = scale.view(*batch_shape, -1)
        return loc, scale

class CriticHead(nn.Module):
    def __init__(self, base_model, n_agents):
        super().__init__()
        self.base_model = base_model
        self.n_agents = n_agents

    def forward(self, obs, adj):
        _, _, values = self.base_model(obs, adj)  # values are [B*N, 1]
        batch_shape = obs.shape[:-1]
        values = values.view(*batch_shape, -1)
        return values


class GAPPO:
    def __init__(self, env, args):
        self.args = args
        self.env = env
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
            backbone = Init_GAPPO(
                action_size=action_size,
                n_agents=n_agents,
                input_features=obs_size,
                hidden_dim=128
            )

            actor_head = ActorHead(backbone, n_agents)

            module = TensorDictModule(
                actor_head, 
                in_keys=[(group, "observation"), ("player", "adjacency")], 
                out_keys=[(group, "loc"), (group, "scale")]
            )

            policy = ProbabilisticActor(
                module=module,
                spec=self.env.input_spec["full_action_spec"][group]["action"],
                in_keys=[(group, "loc"), (group, "scale")],
                out_keys=[(group, "action")],
                distribution_class=TanhNormal,
                distribution_kwargs={
                    "low": self.env.input_spec["full_action_spec"][group]["action"].space.low,
                    "high": self.env.input_spec["full_action_spec"][group]["action"].space.high,
                },
                return_log_prob=True,
                log_prob_key=(group, "sample_log_prob")
            )
            self.policies[group] = policy

            # Critic network
            critic_head = CriticHead(backbone, n_agents)
            critic = TensorDictModule(
                critic_head, 
                in_keys=[(group, "observation"), (group, "adjacency")], 
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


    
