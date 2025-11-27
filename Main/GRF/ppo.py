# Utils
from utils import apply_orthogonal_init
# Torch
import torch

# Training
from train import Train

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

#Logging
from tqdm import tqdm
import time




class PPO(Train):
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.device = args.device
        self.global_step = 0

        self.policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["player", "observation"].shape[-1],  # n_obs_per_agent
        n_agent_outputs=19,  # n_actions_per_agents
        n_agents=args.n_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=args.mappo,  # whether to share parameters across agents
        device=args.device,
        depth=3,
        num_cells=[256, 128, 64],
        activation_class=torch.nn.ReLU,
        ).to(args.device)
        apply_orthogonal_init(self.policy_net)


        policy_module = TensorDictModule(
            self.policy_net,
            in_keys=[("player", "observation")],
            out_keys=[("player", "logits")],
        ).to(args.device)

        self.policy = ProbabilisticActor(
            module=policy_module,
            spec=env.action_spec_unbatched,
            in_keys=[("player", "logits")],
            out_keys=[env.action_key],
            distribution_class=Categorical,
            return_log_prob=True,
        ).to(args.device)  # we'll need the log-prob for the PPO loss


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
        ).to(args.device)
        apply_orthogonal_init(critic_net)

        self.critic = TensorDictModule(
            module=critic_net,
            in_keys=[("player", "observation")],
            out_keys=[("player", "state_value")],
        ).to(args.device)


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
        ).to(args.device)

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

        self.optim = torch.optim.Adam(self.loss_module.parameters(), args.learning_rate)

        super().__init__(args, env, self.collector, self.replay_buffer, self.loss_module, self.optim)  