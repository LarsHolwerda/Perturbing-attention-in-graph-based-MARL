# Perturbing Attention in Graph-Based MARL for Strategic and Diverse Coordination
This repository contains the code accompanying the master thesis by Lars Holwerda at Utrecht University.

**Thesis link:** 
https://studenttheses.uu.nl/handle/20.500.12932/50954

## Project description
The project investigates how diversity can be encouraged in graph-based multi-agent reinforcement learning (MARL).

Experiments are conducted in two environments:

#### Google Research Football (GRF)
<img src="https://github.com/user-attachments/assets/1a183051-f8ad-4336-a89e-0a3c8247c45c" alt="GRF gif" width="600"/>

#### Simple Tag
<img src="https://github.com/user-attachments/assets/7cc51577-0ffa-49f5-a392-afaa8fc782b6" alt="Simple Tag gif" width="600"/>

## Introduction
Graph-based MARL, and in general MARL training, often converges to homogeneous behaviours, where agents converge to similar roles and strategies. 

In this thesis, we investigate how diversity can be encouraged in graph-based Multi-Agent Reinforcement Learning to improve strategic coordination. 

To study this, we compare different training and communication setups that are expected to influence coordination and diversity. In particular, we focus on the following aspects:

- **centralized** versus **decentralized** learning  
- MARL policies **with communication** versus **without communication** 
- **fixed** communication structures versus **perturbed** communication structures  

The following algorithms are evaluated in these comparisons:

**Baselines**

- MAPPO (centralized non-graph-based MARL algorithm)
- IPPO (decentralized non-graph-based MARL algorithm)
- GAPPO (centralized graph-based MARL algorithm)
  
**Proposed methods**
  
- **IGAPPO** (decentralized graph-based MARL algorithm)
- **PGAPPO** (centralized, perturbed graph-based MARL algorithm)
- **PIGAPPO** (decentralized, perturbed graph-based MARL algorithm)

---

## Installation

It is recommended to use a virtual environment (e.g. [Conda](https://www.anaconda.com/)) to avoid dependency conflicts.

### Google Research Football (GRF)
This project uses the PettingZoo wrapper for Google Research Football.

Follow the official installation guide:
https://github.com/xihuai18/gfootball-gymnasium-pettingzoo

#### Required manual patch:
In envs/env_name/lib/python3.10/site-packages/gfootball/gfootball_pettingzoo_v1.py add on line 236 ```score_reward``` to the info dict in the following way:
```python
for agent_id, agent in enumerate(self.agents):
            observation_dict[agent] = observation_array[agent_id]
            info_key2dict[agent] = {
                "score_reward": 0.0
            }
```
This is necessary because if score_reward is not included during a reset, TorchRL will assume that the field does not exist in the environment, which will cause TorchRL to crash with the error: 
```
TypeError: 'NoneType' object does not support item assignment
```

Then install in the virtual environment:
```bash
pip install torchrl
pip install tqdm
```

### Simple Tag
Navigate to the Simple_tag directory and install ```requirements.txt```:
```bash
cd Simple_tag
pip install -r requirements.txt
```

---

## Running Experiments
Training scripts and configurations are provided for both environments in the [GRF config](https://github.com/LarsHolwerda/Perturbing-attention-in-graph-based-MARL/blob/main/Main/GRF/config/config.py) and [Simple Tag config](https://github.com/LarsHolwerda/Perturbing-attention-in-graph-based-MARL/blob/main/Main/Simple_tag/config/config.py). 

Each configuration includes documentation for general training runs, algorithm hyperparameters, perturbation parameters, and evaluation/logging settings.

---

## Evaluation
Training progress and results can be monitored through [wandb](https://wandb.ai/)

Logging can be enabled or disabled, and wandb project and entity names can be configured as needed.  

When enabled, all training metrics and visualizations are automatically tracked. 

### Metrics and Visualizations
The following metrics and visualizations are logged during training:

#### Episode rewards

#### PPO loss components, including:
- policy gradient loss
- value loss
- entropy loss
- approximate KL divergence
- clipping fraction

#### Training durations, such as:
- data collection
- optimization
- GAE computation
- SIPO reward calculation (if applicable)
- policy synchronization

In addition to standard training metrics, the following analysis tools are provided:
#### Diversity metric based on symmetric KL divergence between agent policies

#### Visualization of learned attention weights
  <img width="600" alt="afbeelding" src="https://github.com/user-attachments/assets/1804158c-9912-47c2-aa44-26a280bcabaa" />
  
#### Episode rollout visualizations for qualitative analysis of agent behaviour
<img src="https://github.com/user-attachments/assets/727a9347-ee7c-4632-93f3-499478857987" alt="Beauty rebound gif" width="600"/>




