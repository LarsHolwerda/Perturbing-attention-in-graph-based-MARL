# Logging
import os
import glob
from moviepy import VideoFileClip
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

def compute_behavioral_diversity(subdata):
    logits = subdata.get(("player", "logits"))
    n_agents = logits.shape[1]

    distances = {}
    for i, j in combinations(range(n_agents), 2):
        logits_i = logits[:, i, :]
        logits_j = logits[:, j, :]  

        pi_i = F.log_softmax(logits_i, dim=-1)
        pi_j = F.softmax(logits_j, dim=-1)
        kl_ij = F.kl_div(pi_i, pi_j, reduction="batchmean", log_target=False)

        pi_j_log = F.log_softmax(logits_j, dim=-1)
        pi_i_prob = F.softmax(logits_i, dim=-1)
        kl_ji = F.kl_div(pi_j_log, pi_i_prob, reduction="batchmean", log_target=False)

        symmetric_kl = kl_ij + kl_ji
        distances[f"KL_agent_{i}_{j}"] = symmetric_kl.item()

    return distances


