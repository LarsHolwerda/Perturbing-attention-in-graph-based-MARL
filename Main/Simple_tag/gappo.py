# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

# PyG
from torch_geometric.data import Batch
from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.utils import unbatch
from torch_geometric.utils import softmax
from torch_geometric.typing import OptTensor

# Training
from train import Train

# Utils
from utils import apply_orthogonal_init, create_fully_connected_adj
from torchrl.objectives.common import add_random_module
from typing import Optional

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn import TensorDictSequential
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

# Env
from env import create_env

#Logging
from tqdm import tqdm

def create_geo_batch(m_obs, m_adj, device):
    """
      m_obs: [batch_size, n_agents, embedding_dim] batch of observations of agents
      m_adj: [batch_size, n_agents, n_agents] batch of adjacency matrices
    """
    # Move tensors to device
    m_obs = m_obs.to(device, non_blocking=True) # Gets all observations
    m_adj = m_adj.to(device, non_blocking=True) # Gets all graphs and the edges between them

    B, N, D = m_obs.shape

    # Returns all edges in [B, src, dst] format
    m_adj_flat = m_adj.nonzero(as_tuple=False) 

    # Create edge_index by vectorization and returns [2, num_edges] shape in [[src], [dst]] format
    edge_index = torch.stack([
        m_adj_flat[:, 0] * N + m_adj_flat[:, 1],  # source
        m_adj_flat[:, 0] * N + m_adj_flat[:, 2],  # target
    ])

    # Create mapping for which graph/batch each node belongs to. Example: [0, 0, 1, 1]
    batch = torch.arange(B, device=device).repeat_interleave(N) 

    # Flatten node features to fit Batch structure
    x = m_obs.reshape(B * N, D) 
    
    # Build Batch object
    batch = Batch(x=x, edge_index=edge_index, batch=batch) 
    return batch


class MLPEncoder(nn.Module):
    """
        Encoder: creates embedding of node observations
    """
    def __init__(self, algorithm, input_features, embedding_dim=128):
        super(MLPEncoder, self).__init__()
        if algorithm in ["GAPPO", "PGAPPO"]:
            self.layer1 = nn.Linear(input_features, 256)
            self.layer2 = nn.Linear(256, 128)
            self.layer3 = nn.Linear(128, embedding_dim)
        elif algorithm in ["IGAPPO", "PIGAPPO"]:
            self.layer1 = nn.Linear(input_features, 128)
            self.layer2 = nn.Linear(128, 64)
            self.layer3 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        """
            x: Tensor([batch_size, , n_agents, observation_shape]) observation of an agents

            returns: x: Tensor([batch_size, n_agents, embedding_dim]) embedded observations of agents
        """
        # encode the received observation with MLP
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        # return embedded observation of agent
        return x

class NoisyGATv2Conv(GATv2Conv):
    """
        Addition to GATv2Conv: noise added to attention logits during perturbation periods in PGAPPO / PIGAPPO
    """
    def __init__(self, *args, device, noise_scale=0.0, num_edges=None, window_size=None, num_envs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.noise_scale = noise_scale
        self.num_edges = num_edges
        self.window_size = window_size
        self.num_envs = num_envs
        self.edge_noise = None

    def precomputed_noise(self, edge_index, batch):
        """Precompute per-env noise schedule for the next batch."""
        # Source nodes of edges
        src_nodes = edge_index[0] 

        # Maps each edge to the environment it belongs to and ensure env indices are within num_envs
        env_indices = batch[src_nodes]   
        env_indices = env_indices % self.num_envs

        # Assign a step to each edge
        step_ids = torch.arange(edge_index.size(1), device=self.device) // (self.num_envs * self.num_edges)

        # Assign a window to each edge 
        window_ids = step_ids // self.window_size 

        # Create pairs of (env, window) for each edge
        env_window_pairs = torch.stack([env_indices, window_ids], dim=1) 

        # Get unique (env, window) pairs and their indices
        unique_pairs, unique_indices = torch.unique(env_window_pairs, dim=0, return_inverse=True)
        num_groups = unique_pairs.size(0)

        # Sample noise for each (env, window) pair
        base_noise = torch.randn(num_groups, self.num_edges, device=self.device) * self.noise_scale

        # Get indexes for each edge to retrieve the correct noise 
        local_idx = torch.arange(edge_index.size(1), device=self.device) % self.num_edges

        # Map noise to their edges
        edge_noise = base_noise[unique_indices, local_idx]
        edge_noise = edge_noise.unsqueeze(-1).expand(-1, self.heads)
        self.edge_noise = edge_noise  

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                    index: Tensor, ptr: OptTensor,
                    dim_size: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        if self.edge_noise is not None:
            alpha = alpha + self.edge_noise
        alpha = softmax(alpha, index, ptr, dim_size)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha
    


class GATLayer(nn.Module):
    """
        Graph Attention convolution mechanism: Creates latent features by combining agent's observations with
        observations of neighbors
        Adds noise to the attention logits during perturbation periods in PGAPPO / PIGAPPO
    """
    def __init__(self, device, algorithm, n_agents, args, embedding_dim=128, heads=8):
        super(GATLayer, self).__init__()
        # Initialize GAT-layer
        self.device = device
        self.algorithm = algorithm
        self.noise_scale = args.noise_scale
        self.n_agents = n_agents
        self.num_envs = args.number_of_workers
        self.window_size = args.window_size
        self.num_edges = self.n_agents * self.n_agents
        self.gat_layer = NoisyGATv2Conv(embedding_dim, embedding_dim, heads=heads, concat=False, device=self.device, noise_scale=args.noise_scale, num_edges=self.num_edges, window_size=self.window_size, num_envs=self.num_envs)

    def forward(self, x, edge_index, batch, global_step, args):
        """
            x: Tensor([batch_size, n_agents, embedding_dim]): each agent is seen as a node in the graph with it's embedding as the
                node feature
            edge_index: Tensor(): the topology of the 'x' graph. The nodes of communicating agents are connected with
                each other

            returns: latent_features, att_weights: computed latent features and attention weights
        """
        # create latent features and attention weights
        use_perturbation = False
        if self.algorithm in ["PGAPPO", "PIGAPPO"] and global_step >= args.perturb_attention_start_step:
            cycle_step = (global_step - args.perturb_attention_start_step) % \
                         (args.normal_training_period + args.perturbation_period)
            if cycle_step >= args.normal_training_period:
                use_perturbation = True
        
        # Set noise scale dynamically
        if use_perturbation:
            self.gat_layer.noise_scale = self.noise_scale
            self.gat_layer.precomputed_noise(edge_index, batch)   
        else:
            self.gat_layer.noise_scale = 0.0
            self.gat_layer.edge_noise = None
            
        latent_features, att_weights = self.gat_layer(x, edge_index, return_attention_weights=True)
        self.last_att_weights = att_weights # Store last attention weights
        return latent_features, att_weights

add_random_module(GATLayer)
class ActionLayer(nn.Module):
    """
        Action Layer: computes actions based on created latent features
    """
    def __init__(self, algorithm, actions, embedding_dim=128):
        super(ActionLayer, self).__init__()
        # linear layer computes logits based on the latent features
        if algorithm in ["GAPPO", "PGAPPO"]:
            self.fc1 = nn.Linear(embedding_dim*2, 128)
            self.fc2 = nn.Linear(128, actions)
        elif algorithm in ["IGAPPO", "PIGAPPO"]:
            self.fc1 = nn.Linear(embedding_dim*2, 64)
            self.fc2 = nn.Linear(64, actions)

    def forward(self, i1, i2):
        x = torch.cat([i1, i2], dim=-1)
        x = F.relu(self.fc1(x))
        # compute logits based on latent features
        logits = self.fc2(x)  
        
        # return computed logits
        return logits


class Init_GAPPO(nn.Module):
    def __init__(self, algorithm, actions, n_agents, input_features, device, global_step, args, hidden_dim=128):
        super().__init__()

        self.n_agents = n_agents
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.actions = actions
        self.device = device
        self.adj = create_fully_connected_adj(self.n_agents, self.device)
        self.algorithm = algorithm
        self.global_step = global_step
        self.args = args

        # Layers
        self.encoder = MLPEncoder(self.algorithm, input_features=self.input_features, embedding_dim=self.hidden_dim).to(self.device)
        self.gat = GATLayer(self.device, self.algorithm, self.n_agents, self.args, embedding_dim=self.hidden_dim, heads=8).to(self.device)
        self.action_layer = ActionLayer(self.algorithm, actions=actions, embedding_dim=self.hidden_dim).to(self.device)
        self.value_proc = lambda i1, i2: torch.cat([i1, i2], dim=-1)
        if self.algorithm in ["GAPPO", "PGAPPO"]:
            self.value_branch = nn.Sequential(
                nn.Linear(self.hidden_dim*2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ).to(self.device)
        elif self.algorithm in ["IGAPPO", "PIGAPPO"]:
            self.value_branch = nn.Sequential(
                nn.Linear(self.hidden_dim*2, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            ).to(self.device)


    def forward(self, global_obs, enc_by_backbone=None, agent_index=None):
        """
        Inputs:
            global_obs: Tensor [B, N, obs_dim]
            adj:        Tensor [B, N, N]

        Returns:
            logits:     Tensor [B * N, 1] to avoid a batch dimension mismatch when recording videos
            values:     Tensor [B * N, 1]
        """
        
        # ensure shape [B, N, obs_dim] is met (necessary for parallel envs)
        if global_obs.dim() == 2:
            global_obs = global_obs.unsqueeze(0)

        if global_obs.dim() == 4:  # [B, T, N, obs_dim]
            B, T, N, obs_dim = global_obs.shape
            global_obs = global_obs.view(B * T, N, obs_dim)  # [B*T, N, obs_dim]

        B, N, obs_dim = global_obs.shape

        # 1. Encode observations using MLP encoder
        if self.algorithm in ["GAPPO", "PGAPPO"]:
            encoded = self.encoder(global_obs)
        elif self.algorithm in ["IGAPPO", "PIGAPPO"]:
            enc_agents_list = []
            for agent in range(N):
                enc_agent = enc_by_backbone[agent]  # encoding of agent
                if agent != agent_index:
                    enc_agent = enc_agent.detach()  # block gradient of the other agents
                enc_agents_list.append(enc_agent)
            encoded = torch.cat(enc_agents_list, dim=1)  # concatenate all encodings

        # 2. Enable agent communication with GAT-layer
        geo_batch = create_geo_batch(encoded, self.adj.unsqueeze(0).expand(B, -1, -1), device=global_obs.device)
        self.geo_batch = geo_batch
        rel, att_weights = self.gat(geo_batch.x, geo_batch.edge_index, geo_batch.batch, self.global_step, args=self.args)  
        rel_unbatched = torch.stack(unbatch(rel, geo_batch.batch))
        
        # 3. Compute action probabilities based on the latent features
        logits = self.action_layer(encoded, rel_unbatched)  # [B, N, num_actions]
        logits = logits.reshape(-1, self.actions)  # [B * N, num_actions] avoids the batch dimension mismatch when recording videos
        values = self.value_branch(self.value_proc(encoded, rel_unbatched))  # [B, N, 1]
        values = values.reshape(-1, 1) # [B * N, 1] avoids the batch dimension mismatch when recording videos
        return logits, values

class ActorHead(nn.Module):
    '''
        Reshapes the output of the shared base model to [B, N, num_actions]. 
        We need this wrapper around the base model because the loss function expects a different network for the actor and critic.
    '''
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, obs):
        logits, _ = self.base_model(obs)
        batch_shape = obs.shape[:-1]
        # Reshape logits from B*N, num_actions to B, N, num_actions
        logits = logits.view(*batch_shape, -1)
        
        return logits

class IndependentActorHead(nn.Module):
    '''
        Reshapes the output of the independent base model to [B, N, num_actions]. 
        We need this wrapper around the base model because the loss function expects a different network for the actor and critic.
        Encodings are per agent, from their own network. Logits are also per agent, and from their own network.
    '''
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, obs):
        add_batch = False # If video recording, we need to add a batch dimension    
        if obs.ndim == 2:  # [N, obs_dim]
            obs = obs.unsqueeze(0)  # -> [1, N, obs_dim]
            add_batch = True 

        B, N, _ = obs.shape
        
        # Encode observations using encoder of each backbone
        enc_by_backbone = []
        for agent, net in enumerate(self.base_model):
            obs_agent = obs[:, agent:agent+1, :] # Use only the observation of the specific agent to get encoding from current backbone
            enc_agent = net.encoder(obs_agent)         
            enc_by_backbone.append(enc_agent)

        logits = []
        # Loop over agents and their network to get individual logits
        for agent, network in enumerate(self.base_model):
            logits_agent, _ = network(obs, enc_by_backbone, agent)
            logits_agent = logits_agent.view(B, N, -1)
            logits.append(logits_agent[:, agent:agent+1])  # Append only the logits of the specific agent, i:i+1 keeps the agent dimension [B, 1, actions]
        # Concatenate logits of all agents along the agent dimension [B, N, actions]
        logits = torch.cat(logits, dim=1)

        if add_batch:
            logits = logits.squeeze(0)  
        
        return logits
    

class CriticHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, obs):
        if obs.dim() == 4:  # [B, T, N, obs_dim]
            B, T, N, obs_dim = obs.shape
            obs = obs.view(B * T, N, obs_dim)  # [B*T, N, obs_dim]
            batch_shape = (B, T, N)
        else:
            batch_shape = obs.shape[:-1]
            B, N, _ = obs.shape
        _, values = self.base_model(obs)  
        return values.view(*batch_shape, 1)

class IndependentCriticHead(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, obs):
        has_time_dim = obs.dim() == 4  
        if has_time_dim:  # [B, T, N, obs_dim]
            B, T, N, obs_dim = obs.shape
            obs = obs.view(B * T, N, obs_dim)  # [B*T, N, obs_dim]
        else:
            B, N, _ = obs.shape
        
        enc_by_backbone = []
        for agent, net in enumerate(self.base_model):
            obs_agent = obs[:, agent:agent+1, :] 
            enc_agent = net.encoder(obs_agent)         
            enc_by_backbone.append(enc_agent)
        values = []

        # Loop over agents and their network to get individual values
        for agent, network in enumerate(self.base_model):
            _, value_agent = network(obs, enc_by_backbone, agent)     # [B, 1, 1] Use only the observation of the specific agent to get value from backbone
            value_agent = value_agent.view(-1, N, 1)
            values.append(value_agent[:, agent:agent+1])  # Append only the value of the specific agent, agent:agent+1 keeps the agent dimension [B, 1, 1]
        # Concatenate values of all agents along the agent dimension [B, N, 1]
        values = torch.cat(values, dim=1)
        if has_time_dim:  # has T
            values = values.view(B, T, N, 1)
        else:
            values = values.view(B, N, 1)
        return values

class GAPPO(Train):
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.global_step = 0
        self.agent_frozen = False
        self.policies = {}
        self.critics = {}
        self.losses = {}
        self.optimizers = {}
        
        # Create a single instance of your env for specs
        temp_env = create_env()
        self.group_map = temp_env.group_map

        # Loop over groups to create separate policy, critic, loss and optimizer for each group
        for group, agents in self.group_map.items():
            n_agents = len(agents)
            obs_size = temp_env.observation_spec[group, "observation"].shape[-1]
            actions = temp_env.input_spec["full_action_spec"][group]["action"].n  

            # Policy network
            if args.shared_backbone:
                # Shared backbone
                backbone = Init_GAPPO(
                    algorithm=args.algorithm,
                    actions=actions,
                    n_agents=n_agents,
                    input_features=obs_size,
                    device=args.device,
                    global_step=self.global_step,
                    args=args,
                    hidden_dim=128                
                ).to(args.device)
                apply_orthogonal_init(backbone)
            
                actor_head = ActorHead(backbone)
            else:
                backbones = nn.ModuleList([
                    Init_GAPPO(
                        algorithm=args.algorithm,
                        actions=actions,
                        n_agents=n_agents,              # each backbone sees only its own obs
                        input_features=obs_size,
                        device=args.device,
                        global_step=self.global_step,
                        args=args,
                        hidden_dim=64
                    ).to(args.device)
                    for _ in range(n_agents)
                ])
                for b in backbones: apply_orthogonal_init(b)

                actor_head = IndependentActorHead(backbones)

            # Wrap in tensordict module
            module = TensorDictModule(
                actor_head, 
                in_keys=[(group, "observation")], 
                out_keys=[(group, "logits")]
            )

            # Produces actions given the logits
            policy = ProbabilisticActor(
                module=module,
                spec=self.env.input_spec["full_action_spec"][group]["action"],
                in_keys=[(group, "logits")],
                out_keys=[(group, "action")],
                distribution_class=Categorical,
                return_log_prob=True,
                log_prob_key=(group, "sample_log_prob")
            )
            # Append to policies dict
            self.policies[group] = policy

            # Critic network
            if args.shared_backbone:
                critic_head = CriticHead(backbone)
            else:
                critic_head = IndependentCriticHead(backbones)
            # Wrap in tensordict module
            critic = TensorDictModule(
                critic_head, 
                in_keys=[(group, "observation")], 
                out_keys=[(group, "state_value")]
            )
            # Append to critics dict
            self.critics[group] = critic

            # Agent loss module
            loss_module = ClipPPOLoss(
                actor_network=policy,
                critic_network=critic,
                clip_epsilon=args.clip_epsilon,
                entropy_coeff=args.entropy_eps,
                normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
            )
            # We have to tell the loss where to find the keys
            loss_module.set_keys(
                reward=(group, "reward"),
                action=(group, "action"),
                value=(group, "state_value"),
                sample_log_prob=(group, "sample_log_prob"),  
                value_target=(group, "value_target"),  
                # The 'done' and 'terminated' keys will be expanded to match the reward shape
                done=(group, "done"),
                terminated=(group, "terminated"),
                advantage=(group, "advantage"),  
            )
            # Enables GAE inside the PPO loss module
            loss_module.make_value_estimator(
                ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lmbda
            )  
            # Append to losses dict
            self.losses[group] = loss_module
            # Initialize optimizer
            self.optimizers[group] = torch.optim.Adam(loss_module.parameters(), args.learning_rate, fused=True if torch.cuda.is_available() else False)

        # Close the temporary env
        temp_env.close() 

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

        # Give the Train class the components it needs to train with Gappo
        super().__init__(args, env, self.collector, self.replay_buffer, self.losses, self.optimizers, self.group_map)
