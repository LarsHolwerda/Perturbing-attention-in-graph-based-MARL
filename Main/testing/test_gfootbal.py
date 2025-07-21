from gfootball.env import create_environment
from learning_algorithms import RandomAgent
import time

env = create_environment(env_name='academy_3_vs_1_with_keeper', representation='simple115v2', number_of_left_players_agent_controls=3, render=True)
number_of_agents = 3
actions_agents = env.action_space.nvec

agents = [RandomAgent(actions_agents[i]) for i in range(number_of_agents)]

obs, info = env.reset()
done = False
while not done:
    actions = [agents[i].act(obs[i]) for i in range(number_of_agents)]
    print(f"Actions: {actions}")
    next_obs, reward, terminated, truncated, info = env.step(actions)
    print(reward)
    obs = next_obs
    time.sleep(0.10)  


