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
import numpy as np
from itertools import combinations
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Initialize orthogonal weights for layers
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
            elif layer.out_features == 5:
                std = 0.01
            else:
                std = np.sqrt(2)  
            layer_init(layer, std=std, bias_const=0.01)



# Logging
def log_metrics(metrics, step, use_wandb):
    if use_wandb:
        wandb.log(metrics, step=step)
        
def record_video(multi_agent_policy, algorithm, n_agents, n_adv, device, video_path="traces/video.mp4", max_steps=250):
    env = create_render_env()
    td = env.reset()
    multi_agent_policy.to(device)
    multi_agent_policy.eval()
    if algorithm == "IGAPPO":
        policy = multi_agent_policy[0]  # ProbabilisticActor
        actor_head = policy.module[0].module  # TensorDictModule -> ActorHead
        base_models = getattr(actor_head, "base_models", [actor_head.base_model])  
        init_gappos = list(base_models[0])  # list of Init_GAPPO
        agent_att_frames = [[] for _ in range(len(init_gappos))]  # one list per agent
    att_frames = []
    frames = []
    step = 0

    while step < max_steps:
        if step == 249:
            print("Reached max steps, stopping recording.")
        all_done = True
        for group in env.group_map.keys():
            done = td.get((group, "done")) if (group, "done") in td.keys(True, True) else td.get("done")
            if not done.all().item():  # at least one not done
                all_done = False
                break
        if all_done:
            break

        with torch.no_grad():
            td_device = td.to(device)
            # policy forward -> only returns actions
            td_actions = multi_agent_policy(td_device)

            if algorithm == "GAPPO":
            # To capture attention we need to access last_att_weights from GAT
                policy = multi_agent_policy[0] # ProbabilisticActor
                actor_head = policy.module[0].module # TensorDictModule -> ActorHead
                base_model = actor_head.base_model # Init_GAPPO
                edge_index, att_values = base_model.gat.last_att_weights # access the last attention weights from the GAT in the base model
                att_frame = att_to_frame(edge_index, att_values, n_agents=n_agents, n_adv=n_adv) # convert attention weights to frames
                att_frames.append(att_frame) 

            if algorithm == "IGAPPO":
            # To capture attention we need to access last_att_weights from GAT
                for agent, init_gappo in enumerate(init_gappos):
                    edge_index, att_values = init_gappo.gat.last_att_weights
                    att_frame = att_to_frame(edge_index, att_values, n_agents=n_agents, n_adv=n_adv)
                    agent_att_frames[agent].append(att_frame)


        # put the actions back into the tensordict
        agent_action = td_actions.get(("agent", "action")).cpu()
        adv_action = td_actions.get(("adversary", "action")).cpu()
        td.set(("agent", "action"), agent_action)
        td.set(("adversary", "action"), adv_action)
        # step the env with the updated CPU td
        td = env.step(td)
        td = td.get("next") 
        if td is None:
            print("td is None, breaking loop")
             
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        step += 1

    env.close()
    
    # Save as MP4
    imageio.mimwrite(video_path, frames, fps=10)
    print(f"Video saved to {video_path}")
    if algorithm == "GAPPO":
        att_path = video_path.replace(".mp4", "_att.mp4")
        imageio.mimwrite(att_path, att_frames, fps=10)
        print(f"Attention video saved to {att_path}")
    # Save attention video if IGAPPO
    if algorithm == "IGAPPO":
        for agent, frames_list in enumerate(agent_att_frames):
            att_path = video_path.replace(".mp4", f"_att_{agent}.mp4")
            imageio.mimwrite(att_path, frames_list, fps=10)
            print(f"Video saved to {att_path}")

def upload_videos_to_wandb(video_dir="./traces", scenario=None, algorithm=None, step=0):
    video_paths = glob.glob(f"{video_dir}/*.mp4")  
    output_name = f"{scenario}__{algorithm}"
    for mp4_path in video_paths:
        try:
            wandb.log(
                {f"{output_name}__{os.path.basename(mp4_path)}": wandb.Video(mp4_path, fps=10, format="mp4")},
                step=step,
            )
            # Optionally delete after upload
            os.remove(mp4_path)

        except Exception as e:
            print(f"Failed to upload or delete {mp4_path}: {e}")


def compute_behavioral_diversity(subdata, n_adversaries):
    logits = subdata.get(("adversary", "logits"))

    distances = {}
    # Loop over all unique pairs of adversaries
    for i, j in combinations(range(n_adversaries), 2):
        # Select logits for adversaries i and j
        logits_i = logits[:, i, :]
        logits_j = logits[:, j, :]  
        # Compute KL divergence i to j
        pi_i = F.log_softmax(logits_i, dim=-1)
        pi_j = F.softmax(logits_j, dim=-1)
        kl_ij = F.kl_div(pi_i, pi_j, reduction="batchmean", log_target=False)
        # Compute KL divergence j to i
        pi_j_log = F.log_softmax(logits_j, dim=-1)
        pi_i_prob = F.softmax(logits_i, dim=-1)
        kl_ji = F.kl_div(pi_j_log, pi_i_prob, reduction="batchmean", log_target=False)
        # Symmetric KL divergence
        symmetric_kl = kl_ij + kl_ji
        distances[f"KL_agent_{i}_{j}"] = symmetric_kl.item()

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