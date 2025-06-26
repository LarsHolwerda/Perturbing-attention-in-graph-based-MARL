import torch
from gfootball import gfootball_pettingzoo_v1
from torchrl.envs import PettingZooWrapper
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from tensordict import TensorDict
from torchrl.envs import RewardSum, TransformedEnv
from gfootball.env import create_environment
import numpy as np
torch.manual_seed(0)

n_agents = 3
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")

render_env = create_environment(
    env_name='academy_3_vs_1_with_keeper',
    representation='simplev1',
    render=True,
    number_of_left_players_agent_controls=3,
    write_full_episode_dumps=True,
    write_video=True,
    logdir="./traces",
)


raw_env = gfootball_pettingzoo_v1.parallel_env(
    env_name='academy_3_vs_1_with_keeper',
    representation='simplev1',
    number_of_left_players_agent_controls=n_agents,
)
env = PettingZooWrapper(raw_env, group_map=None)

env = TransformedEnv(
    env,
    RewardSum(in_keys=[env.reward_key], out_keys=[("player", "episode_reward")]),
)



share_parameters_policy = False

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
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
        n_agent_inputs=env.observation_spec["player", "observation"].shape[
            -1
        ],  # n_obs_per_agent
        n_agent_outputs=19,  # n_actions_per_agents
        n_agents=n_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=share_parameters_policy,
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
    spec=env.action_spec_unbatched,
    in_keys=[("player", "logits")],
    out_keys=[env.action_key],
    distribution_class=Categorical,
    return_log_prob=True,
)

policy.load_state_dict(torch.load("trained_policies/ippo_policy_fully_trained.pt", map_location=device))
policy.eval()


# Render the environment and policy
for i in range(5):
    obs, info = render_env.reset()
    done = False
    while not done:
        obs_tensordict = TensorDict(
            {
                ("player", "observation"): torch.tensor(np.array(obs), dtype=torch.float32, device=device),
            },
            batch_size=[3],
            device=device,
        )

        with torch.no_grad():
            action_tensordict = policy(obs_tensordict)
        actions = action_tensordict[env.action_key].cpu().numpy()
        actions_list = actions.tolist()

        # Step environment with actions list
        obs, reward, terminated, truncated, info = render_env.step(actions_list)
        done = terminated or truncated

render_env.close()


for layer in policy_net.modules():
    print(layer)
# Check the environment and policy
for i in range(10):    
    counter = 0
    tensordict = env.reset().to(device)

    while True:
        counter += 1

        with torch.no_grad():
            tensordict = policy(tensordict)

        tensordict["player"]["action"] = tensordict["player"]["action"].cpu().detach()

        # Step the environment
        tensordict = env.step(tensordict)
        next_td = tensordict["next"].to(device)

        # Check if the episode is done (global flag)
        if next_td["done"].item():
            print(f"Episode done at step {counter}")
            tensordict = next_td
            break

        # Continue with next timestep
        tensordict = next_td


    episode_reward = tensordict["player", "episode_reward"]
    print(f"Episode Reward per agent: {episode_reward}")
    print(f"Total Episode Reward: {episode_reward.sum().item()}")
env.close()



