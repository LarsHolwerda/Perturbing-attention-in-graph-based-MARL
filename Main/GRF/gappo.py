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
from tensordict import TensorDict
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
    def __init__(self, embedding_dim=128, num_actions=19):
        super(ActionLayer, self).__init__()
        # linear layer computes logits based on the latent features
        self.fc1 = nn.Linear(embedding_dim*2, 128)
        self.fc2 = nn.Linear(128, num_actions)
        apply_orthogonal_init(self)

    def forward(self, i1, i2):
        x = torch.cat([i1, i2], dim=-1)
        x = F.relu(self.fc1(x))
        # compute logits based on latent features
        logits = self.fc2(x)
        # return computed logits
        return logits


class Init_GAPPO(nn.Module):
    def __init__(self, num_outputs, n_agents, input_features, hidden_dim=128, save_weights=False):
        super().__init__()

        self.n_agents = n_agents
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs

        # Layers
        self.encoder = MLPEncoder(input_features=self.input_features, embedding_dim=self.hidden_dim)
        self.gat = GATLayer(embedding_dim=self.hidden_dim, heads=8)
        self.action_layer = ActionLayer(embedding_dim=self.hidden_dim, num_actions=self.num_outputs)
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
        logits = self.action_layer(encoded, rel_unbatched)  # [B, N, num_actions]
        logits = logits.reshape(-1, self.num_outputs)  # [B * N, num_actions]
        values = self.value_branch(self.value_proc(encoded, rel_unbatched))  # [B, N, 1]
        values = values.reshape(-1, 1) # [B * N, 1]
    
        return logits, values

class ActorHead(nn.Module):
    def __init__(self, base_model, n_agents):
        super().__init__()
        self.base_model = base_model
        self.n_agents = n_agents

    def forward(self, obs, adj):
        logits, _ = self.base_model(obs, adj)
        batch_shape = obs.shape[:-1] 
        
        if logits.dim() == 2 and logits.shape[0] == batch_shape.numel():
            logits = logits.view(*batch_shape, -1)
        return logits

class CriticHead(nn.Module):
    def __init__(self, base_model, n_agents):
        super().__init__()
        self.base_model = base_model
        self.n_agents = n_agents

    def forward(self, obs, adj):
        _, values = self.base_model(obs, adj)
        batch_shape = obs.shape[:-1]  
        
        if values.dim() == 2 and values.shape[0] == batch_shape.numel():
            values = values.view(*batch_shape, -1)  
        return values


class GAPPO:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.device = args.device
        self.global_step = 0

        self.base_model = Init_GAPPO(
            num_outputs=19, 
            n_agents=args.n_agents,
            input_features=env.observation_spec["player", "observation"].shape[-1],
        )

        policy_module = TensorDictModule(
            module=ActorHead(self.base_model, self.args.n_agents),
            in_keys=[("player", "observation"), ("player", "adjacency")],
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

        critic_module = TensorDictModule(
            module=CriticHead(self.base_model, self.args.n_agents),
            in_keys=[("player", "observation"), ("player", "adjacency")],
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
            critic_network=critic_module,
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
                    # Logging diversity metrics
                    get_diversity_metrics = compute_behavioral_diversity(subdata)
                    diversity_metrics = {f"diversity/{k}": v for k, v in get_diversity_metrics.items()}
                    log_metrics(diversity_metrics, step=self.global_step, use_wandb=self.args.track)


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
        torch.save(self.policy.state_dict(), "trained_policies/gappo_policy.pt")

    
