
# Env
from mpe2 import simple_tag_v3
from torchrl.envs.utils import check_env_specs
from torchrl.envs import PettingZooWrapper

# Get command line arguments
from config import parse_training_args  
args = parse_training_args()


# Create Pettingzoo parallel env for training
def create_env():
    base_env = simple_tag_v3.parallel_env(num_good=args.n_good, 
                    num_adversaries=args.n_adversaries, 
                    num_obstacles=args.n_obstacles,
                    max_cycles=args.max_cycles,
                    continuous_actions=args.continuous_actions,
                    dynamic_rescaling=False
                )
    env = PettingZooWrapper(base_env)
    #print(env.observation_spec["agent", "observation"])
    #print(env.input_spec["full_action_spec"])
    check_env_specs(env)
    return env

# Create Pettingzoo parallel env only for rendering
def create_render_env():
    base_render_env = simple_tag_v3.parallel_env(
                    render_mode="rgb_array",
                    num_good=args.n_good, 
                    num_adversaries=args.n_adversaries, 
                    num_obstacles=args.n_obstacles,
                    max_cycles=args.max_cycles,
                    continuous_actions=args.continuous_actions,
                    dynamic_rescaling=False
                )
    render_env = PettingZooWrapper(base_render_env)

    print(render_env.possible_agents)
    return render_env