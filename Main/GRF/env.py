# Import gfootball and pettingzoo
from gfootball.env import create_environment
from gfootball import gfootball_pettingzoo_v1
from torchrl.envs import PettingZooWrapper

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.transforms import DeviceCastTransform
from torchrl.envs.utils import check_env_specs
from torchrl.envs.transforms import Compose

# Get command line arguments
from config.config import parse_training_args  
args = parse_training_args()


# Create Pettingzoo parallel env
def create_env(seed=None):
    
    raw_env = gfootball_pettingzoo_v1.parallel_env(
        args.env_id,
        representation='simplev1', 
        number_of_left_players_agent_controls=args.n_agents,
    ) 
    if seed is not None: 
        raw_env.reset(seed=seed)
    env = PettingZooWrapper(raw_env, group_map=None)

    env = TransformedEnv(
        env,
        Compose(
            RewardSum(
                in_keys=[("player", "reward")],
                out_keys=[("player", "episode_reward")]
            ),
        )
    )

    check_env_specs(env) 

    return env

def create_render_env():
    render_env = create_environment(
    env_name=args.env_id,
    representation='simplev1',
    render=True,
    number_of_left_players_agent_controls=args.n_agents,
    write_full_episode_dumps=True,
    write_video=True,
    logdir="./traces",
    )
    return render_env