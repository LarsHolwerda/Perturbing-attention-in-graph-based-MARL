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

# Env
from env import create_env

#Logging
from tqdm import tqdm
import time


class PPO(Train):
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

        # Create a single instance of your env for specs
        temp_env = create_env()
        self.group_map = temp_env.group_map
        
        # Loop over groups to create separate policy, critic, loss and optimizer for each group
        for group, agents in self.group_map.items():
            n_agents = len(agents)
            obs_size = temp_env.observation_spec[group, "observation"].shape[-1]
            actions = temp_env.input_spec["full_action_spec"][group]["action"].n

            # Policy network
            backbone = MultiAgentMLP(
                    n_agent_inputs=obs_size,
                    n_agent_outputs=actions,
                    n_agents=n_agents,
                    centralised=False,
                    share_params=mappo,
                    depth=3,
                    num_cells=[256, 128, 64],
                    device=args.device,
                    activation_class=torch.nn.Tanh,
            )
            apply_orthogonal_init(backbone)

            # Wrap in tensordict module
            module = TensorDictModule(
                backbone, 
                in_keys=[(group, "observation")], 
                out_keys=[(group, "logits")]
            )

            # Produces actions given the logits
            policy = ProbabilisticActor(
                module=module,
                spec=env.input_spec["full_action_spec"][group]["action"],
                in_keys=[(group, "logits")],
                out_keys=[(group, "action")],
                distribution_class=Categorical,
                return_log_prob=True,
                log_prob_key=(group, "sample_log_prob"),
            )
            # Append to policies dict
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

            # Wrap in tensordict module
            critic = TensorDictModule(
                critic_net, 
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
                entropy_coef=args.entropy_eps,
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
            self.optimizers[group] = torch.optim.Adam(loss_module.parameters(), args.learning_rate)
        
        # Close the temporary env
        temp_env.close()  

        # Combine all policies into one sequential module
        self.collect_policy = TensorDictSequential(*self.policies.values())
        # Data collector
        self.collector = SyncDataCollector(
            env,
            self.collect_policy.to(args.device),
            device=args.device,
            storing_device=args.device,
            frames_per_batch=args.env_steps_per_batch,
            total_frames=args.total_env_steps,
        )

        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(
                args.env_steps_per_batch, device=args.device
            ),  # We store the env_steps_per_batch collected at each iteration
            sampler=SamplerWithoutReplacement(),
            batch_size=args.minibatch_size,  # We will sample minibatches of this size
        )
        
        super().__init__(args, env, self.collector, self.replay_buffer, self.losses, self.optimizers, self.group_map)  

