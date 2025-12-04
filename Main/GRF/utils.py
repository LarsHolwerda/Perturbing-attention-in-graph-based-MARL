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
from moviepy import VideoFileClip
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


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

# Logging
def log_metrics(metrics, step, use_wandb):
    if use_wandb:
        wandb.log(metrics, step=step)
        

def record_video(env, policy, device, num_episodes=1):
    for i in range(num_episodes):
        # Create render env
        render_env = create_render_env()
        # Reset the environment and get initial observations
        obs, info = render_env.reset()
        done = False

        # Loop through the environment until done
        while not done:
            # Convert observations to tensordict
            obs_tensordict = TensorDict(
                {
                    ("player", "observation"): torch.tensor(np.array(obs), dtype=torch.float32, device=device),
                },
                batch_size=[3],
                device=device,
            )
            # Get actions from the policy
            with torch.no_grad():
                action_tensordict = policy(obs_tensordict)
            actions = action_tensordict[env.action_key].cpu().numpy()
            actions_list = actions.tolist()
            # Step environment with actions list
            obs, reward, terminated, truncated, info = render_env.step(actions_list)
            done = terminated or truncated        

    render_env.close()    

def convert_avi_to_mp4(avi_path, output_name=None):
    mp4_path = os.path.join(os.path.dirname(avi_path), output_name + ".mp4")    
    clip = VideoFileClip(avi_path)
    clip.write_videofile(mp4_path)
    clip.close()
    return mp4_path

def upload_videos_to_wandb(video_dir="./traces", scenario=None, algorithm=None, step=0):
    video_paths = glob.glob(f"{video_dir}/*.avi")
    output_name = f"{scenario}__{algorithm}__step_{step}"
    for avi_path in video_paths:
        try:
            # Convert AVI to MP4
            mp4_path = convert_avi_to_mp4(avi_path, output_name)
            # Upload video to wandb
            wandb.log({f"video_{os.path.basename(mp4_path)}": wandb.Video(mp4_path, fps=30, format="mp4")}, step=step)
            # delete the video after uploading
            os.remove(mp4_path)  
            os.remove(avi_path)
            dump_path = avi_path.replace(".avi", ".dump")                
            os.remove(dump_path)

        except Exception as e:
            print(f"Failed to upload or delete {mp4_path}: {e}")

def compute_behavioral_diversity(tensordict_data, policy, n_agents):
    obs = tensordict_data.get(("player", "observation"))  # (num_envs, batch, n_agents, obs_dim)
    # Get the observation of the first agent in that environment
    obs_one_agent = obs[:, :, 0, :] 
    # Adds an agent dimension with the same observation for all agents
    same_obs_all_agents = obs_one_agent.unsqueeze(2).repeat(1, 1, n_agents, 1)  
    distances = {}

    with torch.no_grad():
        # Get the output of all agents given the same observations
        output_policy = policy[0](same_obs_all_agents)
        # Get only the logits tensor
        if output_policy.ndim == 4:
            logits_tensor = output_policy[0]
        else:
            logits_tensor = output_policy  
        
        # Compute pairwise symmetric KL
        for i, j in combinations(range(n_agents), 2):
            logits_i = logits_tensor[:, i, :]
            logits_j = logits_tensor[:, j, :]

            log_probs_i = F.log_softmax(logits_i, dim=-1)
            log_probs_j = F.log_softmax(logits_j, dim=-1)
            probs_i = log_probs_i.exp()
            probs_j = log_probs_j.exp()

            kl_ij = F.kl_div(log_probs_i, probs_j, reduction="batchmean", log_target=False)
            kl_ji = F.kl_div(log_probs_j, probs_i, reduction="batchmean", log_target=False)
            symmetric_kl = 0.5 * (kl_ij + kl_ji)
            distances[f"diversity/KL_agent_{i}_{j}"] = symmetric_kl.item()

    return distances


# Gappo
def create_fully_connected_adj(n_agents, device):
    adj = torch.ones(n_agents, n_agents, device=device)
    return adj

def coo_to_dense_weights(edge_index, att_weights, n_agents):
    att_values = att_weights.mean(-1).cpu().numpy() # average over heads
    dense_weights = np.zeros((n_agents, n_agents)) # create a dense matrix
    for idx, (src, dst) in enumerate(edge_index.T.cpu().numpy()): # iterate over transposed edges ([2, num_edges] -> [num_edges, 2])
        dense_weights[src, dst] = att_values[idx] # fill in the attention value for each transposed edge
    return dense_weights

def coo_to_dense_weights_batched(edge_index, att_values, node_batch, n_agents, num_envs):
    att_values = att_values.mean(-1)  # average over heads
    src, dst = edge_index # Source and destination nodes of edges

    env_node_counts = torch.bincount(node_batch) # Number of nodes per env
    nodes_per_env = env_node_counts.view(num_envs, -1).sum(dim=1)  # Total nodes per env
    steps_per_env = (nodes_per_env // n_agents).tolist()  # Steps per env

    # Determine max steps for padding
    max_steps = max(steps_per_env) 

    dense_rollout = torch.zeros(num_envs, max_steps, n_agents, n_agents, device=att_values.device) # Create tensor with shape: (num_envs, max_steps, n_agents, n_agents)

    start_idx = 0 # Where does this environment's nodes start in the global indexing
    # Per environment
    for env_idx in range(num_envs):
        n_steps = steps_per_env[env_idx]
        # Steps in this environment
        for step in range(n_steps):
            node_start = start_idx + step * n_agents # Start node index for this step
            node_end = node_start + n_agents # End node index for this step
            mask = (src >= node_start) & (src < node_end) # Mask for edges for nodes in this step
            
            # Create local indexing for nodes in this step
            src_t = src[mask] - node_start
            dst_t = dst[mask] - node_start 
            att_t = att_values[mask] # Attention values for edges of relevant nodes in this step

            dense_rollout[env_idx, step, src_t, dst_t] = att_t # Fill in the dense matrix for this step
        start_idx += n_steps * n_agents # Start index for next step

    return dense_rollout 

# Convert attention weights to a rgb image of the adversary × adversary attention matrix
def att_to_frame(edge_index, att_values, n_agents, n_adv):
    dense_att = coo_to_dense_weights(edge_index, att_values, n_agents)

    # we need only a adversary × adversary matrix
    adv_dense_att = dense_att[:n_adv, :n_adv]

    fig, ax = plt.subplots()
    im = ax.imshow(adv_dense_att, cmap="viridis", vmin=0, vmax=1)
    # Add text annotations
    for i in range(n_adv):
        for j in range(n_adv):
            value = adv_dense_att[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                color="white" if value <= 0.7 else "black"
            )

    # Axis labels
    ax.set_xticks(np.arange(n_adv))
    ax.set_yticks(np.arange(n_adv))
    ax.set_xticklabels([f"adversary_{i}" for i in range(n_adv)])
    ax.set_yticklabels([f"adversary_{i}" for i in range(n_adv)])
    ax.set_xlabel("From adversary")
    ax.set_ylabel("To adversary")
    ax.set_title("Importance of adversary's communicated observation")

    plt.tight_layout()
    plt.close(fig)

    # Convert figure → frame
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)

