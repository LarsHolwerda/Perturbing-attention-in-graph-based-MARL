# Hierarchical graph-based MARL for strategic and diverse coordination
Run environment:
wsl -d Ubuntu-20.04
cd /mnt/c/Users/lardy/Documents/universiteit/BusinessInformaticsjaar1/Thesis/Project/Hierarchical-graph-based-MARL-for-strategic-and-diverse-coordination/Main
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 
conda activate minimal_thesis

Install guide:
follow the install guide on https://github.com/xihuai18/gfootball-gymnasium-pettingzoo
In envs/env_name/lib/python3.10/site-packages/gfootball/gfootball_pettingzoo_v1.py add on line 236 score_reward to the info dict in the following way:
for agent_id, agent in enumerate(self.agents):
            observation_dict[agent] = observation_array[agent_id]
            info_key2dict[agent] = {
                "score_reward": 0.0
            }

This is necessary because if score_reward is not included during a reset, torchrl will assume that the field does not exist in the environment. This will the error: TypeError: 'NoneType' object does not support item assignment

In the conda env install:
pip install torchrl
pip install tqdm


