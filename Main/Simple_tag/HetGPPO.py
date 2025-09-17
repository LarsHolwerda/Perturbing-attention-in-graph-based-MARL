from typing import Optional

import torch
import torch_geometric
from torch import Tensor
from torch import nn
from torch_geometric.nn import MessagePassing, GINEConv, GraphConv, GATv2Conv
from torch_geometric.transforms import BaseTransform


def get_activation_fn(name: Optional[str] = None):
    """Returns a framework specific activation function, given a name string.
   
    Args:
        name (Optional[str]): One of "relu" (default), "tanh", "elu",
            "swish", or "linear" (same as None)..

    Returns:
        A framework-specific activtion function. e.g.
        torch.nn.ReLU. None if name in ["linear", None].

    Raises:
        ValueError: If name is an unknown activation function.
    """
    # Already a callable, return as-is.
    if callable(name):
        return name

    # Infer the correct activation function from the string specifier.
    if name in ["linear", None]:
        return None
    if name == "relu":
        return nn.ReLU
    elif name == "tanh":
        return nn.Tanh
    elif name == "elu":
        return nn.ELU

    raise ValueError("Unknown activation ({}) for framework=!".format(name))


def get_edge_index_from_topology(topology_type: str, n_agents: int):
    assert n_agents > 0

    if n_agents == 1:
        edge_index = torch.empty((2, 1)).long()
        edge_index[:, 0] = torch.Tensor([0, 0])
    # Connected to all
    elif topology_type == "full":
        edge_index = torch.empty((2, (n_agents**2))).long()

        index = 0
        for i in range(n_agents):
            for j in range(n_agents):
                edge_index[:, index] = torch.Tensor([j, i])
                index += 1
        assert index == n_agents**2
    # Connected in a ring
    elif topology_type == "ring":
        edge_index = torch.empty((2, n_agents * 2)).long()

        index = 0
        if n_agents > 2:
            for i in range(n_agents - 1):
                edge_index[:, index] = torch.Tensor([i, i + 1])
                index += 1
                edge_index[:, index] = torch.Tensor([i + 1, i])
                index += 1
        edge_index[:, index] = torch.Tensor([n_agents - 1, 0])
        index += 1
        edge_index[:, index] = torch.Tensor([0, n_agents - 1])
        assert index == (n_agents * 2) - 1
    # Connected in a line
    elif topology_type == "line":
        edge_index = torch.empty((2, (n_agents - 2) * 2 + 2)).long()

        index = 0
        for i in range(n_agents - 1):
            edge_index[:, index] = torch.Tensor([i, i + 1])
            index += 1
            edge_index[:, index] = torch.Tensor([i + 1, i])
            index += 1
        assert index == (n_agents - 2) * 2 + 2
    else:
        assert False
    return edge_index



def parse_simplev1(obs, n1, n2):
    """Parse simplev1 observation for one agent into positions/velocities."""
    idx = 0
    
    # ----- SELF -----
    self_pos = obs[idx:idx+2]  # absolute position
    idx += 2
    self_vel = obs[idx:idx+2]  # absolute velocity
    idx += 2
    self_status = obs[idx:idx+2]  # sprinting, dribbling
    idx += 2

    # ----- RELATIVE POSITIONS -----
    rel_pos_teammates = obs[idx: idx + 2*(n1-1)].reshape(n1-1, 2)
    idx += 2*(n1-1)
    rel_pos_opponents = obs[idx: idx + 2*n2].reshape(n2, 2)
    idx += 2*n2
    rel_pos_ball = obs[idx:idx+2]  # relative to ball
    idx += 2

    # ----- ABSOLUTE POSITIONS -----
    abs_pos_teammates = obs[idx: idx + 2*(n1-1)].reshape(n1-1, 2)
    idx += 2*(n1-1)
    abs_vel_teammates = obs[idx: idx + 2*(n1-1)].reshape(n1-1, 2)
    idx += 2*(n1-1)
    abs_pos_opponents = obs[idx: idx + 2*n2].reshape(n2, 2)
    idx += 2*n2
    abs_vel_opponents = obs[idx: idx + 2*n2].reshape(n2, 2)
    idx += 2*n2

    # ----- BALL -----
    ball_pos = obs[idx:idx+3]   # absolute
    idx += 3
    ball_vel = obs[idx:idx+3]   # absolute
    idx += 3

    # The rest is one-hot encodings (ownership, game mode, active player)
    extra = obs[idx:]

    return {
        "self_pos": self_pos,
        "self_vel": self_vel,
        "self_status": self_status,
        "rel_pos_teammates": rel_pos_teammates,
        "rel_pos_opponents": rel_pos_opponents,
        "rel_pos_ball": rel_pos_ball,
        "abs_pos_teammates": abs_pos_teammates,
        "abs_vel_teammates": abs_vel_teammates,
        "abs_pos_opponents": abs_pos_opponents,
        "abs_vel_opponents": abs_vel_opponents,
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "extra": extra
    }


class MatPosConv(MessagePassing):
    propagate_type = {"x": Tensor, "edge_attr": Tensor}

    def __init__(self, in_dim, out_dim, edge_features, **cfg):
        super().__init__(aggr=cfg["aggr"])

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.activation_fn = get_activation_fn(cfg["activation_fn"])

        self.message_encoder = nn.Sequential(
            torch.nn.Linear(self.in_dim + self.edge_features, self.out_dim),
            self.activation_fn(),
            torch.nn.Linear(self.out_dim, self.out_dim),
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        msg = self.message_encoder(torch.cat([x_j, edge_attr], dim=-1))
        return msg


class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, edge_features, **cfg):
        super().__init__()

        gnn_types = {"GraphConv", "GATv2Conv", "GINEConv", "MatPosConv"}
        aggr_types = {"add", "mean", "max"}

        self.aggr = cfg["aggr"]
        self.gnn_type = cfg["gnn_type"]

        assert self.aggr in aggr_types
        assert self.gnn_type in gnn_types

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.edge_features = edge_features
        self.activation_fn = get_activation_fn(cfg["activation_fn"])

        if self.gnn_type == "GraphConv":
            self.gnn = GraphConv(
                self.in_dim,
                self.out_dim,
                aggr=self.aggr,
            )
        elif self.gnn_type == "GATv2Conv":
            # Default adds self loops
            self.gnn = GATv2Conv(
                self.in_dim,
                self.out_dim,
                edge_dim=self.edge_features,
                fill_value=0.0,
                share_weights=True,
                add_self_loops=True,
                aggr=self.aggr,
            )
        elif self.gnn_type == "GINEConv":
            self.gnn = GINEConv(
                nn=nn.Sequential(
                    torch.nn.Linear(self.in_dim, self.out_dim),
                    self.activation_fn(),
                ),
                edge_dim=self.edge_features,
                aggr=self.aggr,
            )
        elif self.gnn_type == "MatPosConv":
            self.gnn = MatPosConv(
                self.in_dim,
                self.out_dim,
                edge_features=self.edge_features,
                **cfg,
            )
        else:
            assert False

    def forward(self, x, edge_index, edge_attr):
        if self.gnn_type == "GraphConv":
            out = self.gnn(x, edge_index)
        elif (
            self.gnn_type == "GATv2Conv"
            or self.gnn_type == "GINEConv"
            or self.gnn_type == "MatPosConv"
        ):
            out = self.gnn(x, edge_index, edge_attr)
        else:
            assert False

        return out


class GPPOBranch(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        edge_features,
        n_agents,
        centralised,
        edge_index,
        comm_radius_processed,
        out_features_2=None,
        **cfg,
    ):
        super().__init__()

        self.n_agents = n_agents

        self.in_features = in_features
        self.edge_features = edge_features
        self.out_features = out_features
        self.out_features2 = out_features_2
        self.double_output = out_features_2 is not None
        self.centralised = centralised
        self.edge_index = edge_index
        self.comm_radius = comm_radius_processed

        self.hidden_size = 128

        self.activation_fn = get_activation_fn(cfg["activation_fn"])

        self.hetero_encoders = cfg["heterogeneous"]
        self.hetero_gnns = cfg["heterogeneous"]
        self.hetero_decoders = cfg["heterogeneous"]

        if self.centralised:
            # Will not get edge features
            self.centralised_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        torch.nn.Linear(
                            self.in_features * self.n_agents,
                            256,
                        ),
                        self.activation_fn(),
                        torch.nn.Linear(
                            256,
                            self.hidden_size * self.n_agents,
                        ),
                    )
                    for _ in range(self.n_agents if self.hetero_gnns else 1)
                ]
            )
            self.gnns = None
        else:
            self.gnns = nn.ModuleList(
                [
                    GNN(
                        in_dim=self.in_features,
                        out_dim=self.hidden_size,
                        edge_features=self.edge_features,
                        **cfg,
                    )
                    for _ in range(self.n_agents if self.hetero_gnns else 1)
                ]
            )
            self.centralised_mlps = None

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        self.in_features + self.hidden_size, self.hidden_size
                    ),
                    self.activation_fn(),
                    torch.nn.Linear(self.hidden_size, self.hidden_size),
                    self.activation_fn(),
                )
                for _ in range(self.n_agents if self.hetero_decoders else 1)
            ]
        )

        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(self.hidden_size, self.out_features),
                )
                for _ in range(self.n_agents if self.hetero_decoders else 1)
            ]
        )
        if self.double_output:
            self.heads2 = nn.ModuleList(
                [
                    nn.Sequential(
                        torch.nn.Linear(self.hidden_size, self.out_features2),
                    )
                    for _ in range(self.n_agents if self.hetero_decoders else 1)
                ]
            )
        self.share_init_hetero_networks()

    def forward(self, obs, pos, vel):
        batch_size = obs.shape[0]
        device = obs.device
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)

        if self.centralised:
            embedding = obs.view(batch_size, self.n_agents * self.in_features)

            if self.hetero_gnns:
                embedding = torch.stack(
                    [
                        centralised_mlp(embedding).view(
                            batch_size,
                            self.n_agents,
                            self.hidden_size,
                        )[:, i]
                        for i, centralised_mlp in enumerate(self.centralised_mlps)
                    ],
                    dim=1,
                )
            else:
                embedding = self.centralised_mlps[0](embedding).view(
                    batch_size,
                    self.n_agents,
                    self.hidden_size,
                )

        else:
            graph_dict = parse_simplev1(
                x=obs,
                pos=pos,
                vel=vel,
                edge_index=self.edge_index,
                comm_radius=self.comm_radius,
            )

            if self.hetero_gnns:
                embedding = torch.stack(
                    [
                        gnn(graph_dict["x"],
                            graph_dict["edge_index"],
                            graph_dict["edge_attr"],
                            ).view(
                                batch_size,
                                self.n_agents,
                                self.hidden_size,
                            )[:, i]
                        for i, gnn in enumerate(self.gnns)
                    ],
                    dim=1,
                )

            else:
                embedding = self.gnns[0](
                    graph_dict["x"],
                    graph_dict["edge_index"],
                    graph_dict["edge_attr"],
                ).view(batch_size, self.n_agents, self.hidden_size)

        if self.hetero_decoders:
            embedding = torch.stack(
                [
                    decoder(torch.cat([obs[:, i], embedding[:, i]], dim=-1))
                    for i, decoder in enumerate(self.decoders)
                ],
                dim=1,
            )
        else:
            embedding = self.decoders[0](torch.cat([obs, embedding], dim=-1))

        if self.hetero_decoders:
            out = torch.stack(
                [head(embedding[:, i]) for i, head in enumerate(self.heads)],
                dim=1,
            )
            if self.double_output:
                out2 = torch.stack(
                    [head2(embedding[:, i]) for i, head2 in enumerate(self.heads2)],
                    dim=1,
                )
        else:
            out = self.heads[0](embedding)
            if self.double_output:
                out2 = self.heads2[0](embedding)

        return out, (out2 if self.double_output else None)

    def share_init_hetero_networks(self):
        for child in self.children():
            assert isinstance(child, nn.ModuleList)
            for agent_index, agent_model in enumerate(child.children()):
                if agent_index == 0:
                    state_dict = agent_model.state_dict()
                else:
                    agent_model.load_state_dict(state_dict)




# Utils
from utils import apply_orthogonal_init, log_metrics, record_video, upload_videos_to_wandb, compute_behavioral_diversity

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





class HetGPPO:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.device = args.device
        self.global_step = 0

        policy_net = GPPOBranch(
            in_features=env.observation_spec["player", "observation"].shape[-1],
            out_features=19,
            edge_features=d,
            n_agents=args.n_agents,
            centralised=False,
            edge_index=get_edge_index_from_topology("full", args.n_agents),
            comm_radius_processed=None,
            activation_fn="relu",
            gnn_type="MatPosConv",
            aggr="mean",
            heterogeneous=True,
        )


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
        torch.save(self.policy.state_dict(), "trained_policies/hetgppo_policy.pt")
         