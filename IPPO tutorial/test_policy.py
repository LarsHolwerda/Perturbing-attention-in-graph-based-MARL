import torch
from gfootball import gfootball_pettingzoo_v1
from torchrl.envs import PettingZooWrapper
from torchrl.modules import MultiAgentMLP, ProbabilisticActor
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from tensordict import TensorDict
from torchrl.envs import RewardSum, TransformedEnv
from gfootball.env import create_environment

n_agents = 3
device = torch.device("cuda" if not torch.cuda.is_available() else "cpu")

render_env = create_environment(
    env_name='academy_3_vs_1_with_keeper',
    representation='simple115v2',
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

obs_spec = env.observation_spec["player", "observation"]
action_spec = env.action_spec_unbatched

policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["player", "observation"].shape[
            -1
        ],  # n_obs_per_agent
        n_agent_outputs=env.full_action_spec[env.action_key].shape[-1],  # n_actions_per_agents
        n_agents=n_agents,
        centralised=False,  # the policies are decentralised (ie each agent will act from its observation)
        share_params=False,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.ReLU,
)

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
)  # we'll need the log-prob for the PPO loss

policy.load_state_dict(torch.load("trained_policies/ippo_policy.pt", map_location=device))

policy.eval()

# Render the environment and policy
obs = render_env.reset()
done = False

while not done:
    obs_array = obs[0]
    obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device)
    obs_tensordict = TensorDict(
        {("player", "observation"): obs_tensor},
        batch_size=[3],
    )

    with torch.no_grad():
        action_tensordict = policy(obs_tensordict)
    actions = action_tensordict[env.action_key].cpu().numpy()
    actions_list = actions.tolist()

    # Step environment with actions list
    obs, reward, terminated, truncated, info = render_env.step(actions_list)
    done = terminated or truncated

render_env.close()


# # Check the environment and policy
counter = 0
tensordict = env.reset()
tensordict = tensordict.to(device)

while True:
    if tensordict["done"]:
        print(f"Episode done at step {counter}")
        tensordict = env.reset()
        tensordict = tensordict.to(device)
        counter = 0

    counter += 1
    if counter >= 400:
        print("Manual cutoff at step 400")
        tensordict = env.reset()
        counter = 0
        continue  
    with torch.no_grad():
        tensordict = tensordict
        tensordict = policy(tensordict)

    tensordict["player"]["action"] = tensordict["player"]["action"].cpu().detach()
    print("TOP LEVEL keys:", tensordict.keys())  # should include "player"
    print("PLAYER keys:", tensordict["player"].keys())  # should include "action"
    print("action shape:", tensordict["player"]["action"].shape)
    print("action:", tensordict["player"]["action"])
    assert ("player", "action") in tensordict.keys(True), "Missing player/action key"
    assert tensordict["player"]["action"].shape[0] == 3, "Action tensor has wrong shape"
    tensordict = env.step(tensordict)



episode_reward = tensordict["player", "episode_reward"]
print(f"Episode Reward per agent: {episode_reward}")
print(f"Total Episode Reward: {episode_reward.sum().item()}")
env.close()