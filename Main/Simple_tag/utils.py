# Logging
import os
import glob
import wandb
# Torch
import torch
import torch.nn.functional as F
# Environment
from env import create_render_env
# Utils
from tensordict import TensorDict
import numpy as np
from itertools import combinations

# Initialize weights for layers
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        print(f"Initialized layer with shape {layer.weight.shape}, std={std}")
        return layer

def apply_orthogonal_init(model):
    for layer in model.modules():
        if isinstance(layer, torch.nn.Linear):
            if layer.out_features == 1:
                std = 1.0
            elif layer.out_features == 19:
                std = 0.01
            else:
                std = np.sqrt(2)  
            layer_init(layer, std=std, bias_const=0.01)

# Adversary action need to random for training green
def make_pre_step_fill_adv(env, device):
    def pre_step_fill_adv(td):  # td = tensordict from collector
        adv_spec = env.input_spec["full_action_spec"]["adversary"]["action"]
        dist = torch.distributions.Uniform(
            adv_spec.space.low.to(device), adv_spec.space.high.to(device)
        )
        random_actions = dist.sample()
        td.set(("adversary", "action"), random_actions)
        return td
    return pre_step_fill_adv

# Logging
def log_metrics(metrics, step, use_wandb):
    if use_wandb:
        wandb.log(metrics, step=step)
        
import imageio
def record_video(multi_agent_policy, device, video_path="traces/test.mp4", max_steps=250):
    env = create_render_env()
    td = env.reset()
    multi_agent_policy.to(device)
    multi_agent_policy.eval()
    frames = []
    step = 0

    while step < max_steps:
        all_done = True
        for group in env.group_map.keys():
            done = td.get((group, "done")) if (group, "done") in td.keys(True, True) else td.get("done")
            if not done.all().item():  # at least one agent not done
                all_done = False
                break
        if all_done:
            break

        with torch.no_grad():
            # copy obs to device
            td_device = td.to(device)
            # policy forward -> only returns actions
            td_actions = multi_agent_policy(td_device)
            

        # debugging observations and actions
        #for group in env.group_map.keys():
            #for agent in env.group_map[group]:  # agent names in this group
                #print(f"Observation for {agent}: {td.get((group, 'observation'))}")
                #print(f"Action for {agent}: {td_actions.get((group, 'action'))}")
        
        # put the actions back into the *CPU* tensordict
        agent_action = td_actions.get(("agent", "action")).cpu()
        adv_action = td_actions.get(("adversary", "action")).cpu()
        td.set(("agent", "action"), agent_action)
        td.set(("adversary", "action"), adv_action)
        # step the env with the updated CPU td
        td = env.step(td)
        td = td.get("next")  
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        step += 1

    env.close()
        # Save as MP4
    print(f"Captured {len(frames)} frames")
    imageio.mimwrite(video_path, frames, fps=6)
    print(f"Video saved to {video_path}")

def upload_videos_to_wandb(video_dir="./traces", scenario=None, algorithm=None, step=0):
    video_paths = glob.glob(f"{video_dir}/*.mp4")  
    output_name = f"{scenario}__{algorithm}__step_{step}"
    for mp4_path in video_paths:
        try:
            wandb.log(
                {f"video_{os.path.basename(mp4_path)}": wandb.Video(mp4_path, fps=30, format="mp4")},
                step=step,
            )
            # Optionally delete after upload
            os.remove(mp4_path)

        except Exception as e:
            print(f"Failed to upload or delete {mp4_path}: {e}")

def compute_behavioral_diversity(tensordict):
    loc = tensordict.get(("adversary", "loc"))
    scale = tensordict.get(("adversary", "scale"))  
    n_agents = loc.shape[1]

    distances = {}
    for i, j in combinations(range(n_agents), 2):
        # Build Normal distributions for each agent
        dist_i = torch.distributions.Normal(loc[:, i, :], scale[:, i, :])
        dist_j = torch.distributions.Normal(loc[:, j, :], scale[:, j, :])

        # Compute KL divergences
        kl_ij = torch.distributions.kl_divergence(dist_i, dist_j).mean()
        kl_ji = torch.distributions.kl_divergence(dist_j, dist_i).mean()

        # Symmetric KL
        symmetric_kl = kl_ij + kl_ji
        distances[f"KL_agent_{i}_{j}"] = symmetric_kl.item()

    return distances


