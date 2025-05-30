import numpy as np

class RandomAgent:
    def __init__(self, actions_agents):
        self.actions_agents = actions_agents

    def act(self, obs):
        return np.random.randint(self.actions_agents)
