# Import gfootball and pettingzoo
from gfootball.env import create_environment
from gfootball import gfootball_pettingzoo_v1
from torchrl.envs import PettingZooWrapper

# Torch
import torch
import numpy as np

# Tensordict modules
from tensordict.nn import set_composite_lp_aggregate, TensorDictModule
from torch import multiprocessing
from torch.distributions import Categorical

# Data collection
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.transforms import DeviceCastTransform
from torchrl.envs.utils import check_env_specs
from torchrl.envs.transforms import Compose

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor

# Loss
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Utils
torch.manual_seed(0)
from matplotlib import pyplot as plt
from tqdm import tqdm

# Devices
is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if not torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)

import time
if torch.cuda.is_available():
    torch.cuda.synchronize()
  
grf_device = device  # The device where the simulator is run (VMAS can run on GPU)

# Sampling
frames_per_batch = 500  # Number of team frames collected per training iteration
n_iters = 10  # Number of sampling and training iterations
total_frames = frames_per_batch * n_iters

# Training
num_epochs = 15  # Number of optimization steps per training iteration
minibatch_size = 500  # Size of the mini-batches in each optimization step
lr = 5e-4  # Learning rate
max_grad_norm = 10.0  # Maximum norm for the gradients

# PPO
clip_epsilon = 0.2  # clip value for PPO loss
gamma = 0.99  # discount factor
lmbda = 0.95  # lambda for generalised advantage estimation
entropy_eps = 0  # coefficient of the entropy term in the PPO loss

# disable log-prob aggregation
set_composite_lp_aggregate(False).set()

max_steps = 100  # Episode steps before done
num_grf_envs = (
    frames_per_batch // max_steps
)  # Number of vectorized envs. frames_per_batch should be divisible by this number
scenario_name = "grf"
n_agents = 3

def make_env():
    #pettingzoo parallel env
    raw_env = gfootball_pettingzoo_v1.parallel_env(
        'academy_3_vs_1_with_keeper',
        representation='simplev1', 
        number_of_left_players_agent_controls=3,
    ) 
    env = PettingZooWrapper(raw_env, group_map=None)

    env = TransformedEnv(
        env,
        Compose(
            DeviceCastTransform(
                device=device,
                in_keys=[("player", "observation"), ("player", "reward")],
            ),
            RewardSum(
                in_keys=[("player", "reward")],
                out_keys=[("player", "episode_reward")]
            ),
        )
    )
    return env
def main():
    temp_env = make_env()
    reward_key = temp_env.reward_key
    print(temp_env.observation_spec)
    print(temp_env.observation_spec["player", "observation"].shape[
                -1
            ])
    check_env_specs(temp_env)

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        print(f"Initialized layer with shape {layer.weight.shape}, std={std}")
        return layer

    def apply_orthogonal_init(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                if module.out_features == 1:
                    std = 1.0
                elif module.out_features == 19:
                    std = 0.01
                else:
                    std = np.sqrt(2)  
                layer_init(module, std=std, bias_const=0.01)

    policy_net = MultiAgentMLP(
            n_agent_inputs=temp_env.observation_spec["player", "observation"].shape[
                -1
            ],  # n_obs_per_agent
            n_agent_outputs=19,  # n_actions_per_agents
            n_agents=n_agents,
            centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
            share_params=False,
            device=device,
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

    policy = ProbabilisticActor(
        module=policy_module,
        spec=temp_env.action_spec_unbatched,
        in_keys=[("player", "logits")],
        out_keys=[temp_env.action_key],
        distribution_class=Categorical,
        return_log_prob=True,
    )  # we'll need the log-prob for the PPO loss

    critic_net = MultiAgentMLP(
        n_agent_inputs=temp_env.observation_spec["player", "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=n_agents,
        centralised=False,
        share_params=False,
        device=device,
        depth=3,
        num_cells=[256, 128, 64],
        activation_class=torch.nn.ReLU,
    )
    apply_orthogonal_init(critic_net)

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("player", "observation")],
        out_keys=[("player", "state_value")],
    )

    print("Policy and critic networks initialized.")
    collector = MultiSyncDataCollector(
        [make_env for i in range(num_grf_envs)],
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )
    print("We are here")
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,  # We will sample minibatches of this size
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_coef=entropy_eps,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=temp_env.reward_key,
        action=temp_env.action_key,
        value=("player", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("player", "done"),
        terminated=("player", "terminated"),
    )


    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    )  # We build GAE
    GAE = loss_module.value_estimator

    optim = torch.optim.Adam(loss_module.parameters(), lr)

    pbar = tqdm(total=n_iters, desc="episode_reward_mean = 0")

    del temp_env  
    print("Starting collection loop...")
    episode_reward_mean_list = []
    for i, tensordict_data in enumerate(collector):
        print(f"Iteration {i}: collected batch, shape = {tensordict_data.shape}")
        t0 = time.time()  
        tensordict_data = tensordict_data.to(device)

        tensordict_data.set(
            ("next", "player", "done"),
            tensordict_data.get(("next", "done"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", reward_key))),
        )
        tensordict_data.set(
            ("next", "player", "terminated"),
            tensordict_data.get(("next", "terminated"))
            .unsqueeze(-1)
            .expand(tensordict_data.get_item_shape(("next", reward_key))),
        )
        # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)
        gae_start = time.time()
        with torch.no_grad():
            GAE(
                tensordict_data,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )  # Compute GAE and add it to the data
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"[{i}] GAE computation time: {time.time() - gae_start:.3f}s")
        buffer_start = time.time()
        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
        replay_buffer.extend(data_view)
        print(f"[{i}] Replay buffer extend time: {time.time() - buffer_start:.3f}s")

        opt_start = time.time()
        for _ in range(num_epochs):
            for _ in range(frames_per_batch // minibatch_size):
                subdata = replay_buffer.sample()
                subdata = subdata.to(device)
                loss_vals = loss_module(subdata)

                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )

                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), max_grad_norm
                )  # Optional

                optim.step()
                optim.zero_grad()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        print(f"[{i}] Optimization step time: {time.time() - opt_start:.3f}s")
        sync_start = time.time()
        collector.update_policy_weights_()
        print(f"[{i}] Policy weight sync time: {time.time() - sync_start:.3f}s")

        print(f"[{i}] TOTAL ITER TIME: {time.time() - t0:.3f}s\n")


        # Logging
        done = tensordict_data.get(("next", "player", "done"))
        episode_reward_mean = (
            tensordict_data.get(("next", "player", "episode_reward"))[done].mean().item()
        )
        episode_reward_mean_list.append(episode_reward_mean)
        pbar.set_description(f"episode_reward_mean = {episode_reward_mean}", refresh=False)
        pbar.update()
        
    torch.save(policy.state_dict(), "trained_policies/ippo_policy.pt")
        
    plt.plot(episode_reward_mean_list)
    plt.xlabel("Training iterations")
    plt.ylabel("Reward")
    plt.title("Episode reward mean")
    plt.show()
    """
    # render env
    render_env = create_environment(
        env_name='academy_3_vs_1_with_keeper',
        representation='simple115v2',
        render=True,
        number_of_left_players_agent_controls=3,
        write_full_episode_dumps=True,
        write_video=True,
        logdir="./traces",
    )

    # Render the environment
    obs = render_env.reset()
    done = False

    while not done:
        actions = list(render_env.action_space.sample())
        obs, reward, terminated, truncated, info = render_env.step(actions)

    render_env.close()
    """
if __name__ == "__main__":
    main()