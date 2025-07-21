# Import gfootball and pettingzoo
from gfootball import gfootball_pettingzoo_v1
from torchrl.envs import PettingZooWrapper

# Env
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.envs.transforms import DeviceCastTransform
from torchrl.envs.utils import check_env_specs
from torchrl.envs.transforms import Compose

# Get command line arguments
from config import parse_training_args  
args = parse_training_args()


# Create Pettingzoo parallel env
def create_env():
    
    raw_env = gfootball_pettingzoo_v1.parallel_env(
        'academy_3_vs_1_with_keeper',
        representation='simplev1', 
        number_of_left_players_agent_controls=3,
    ) 
    env = PettingZooWrapper(raw_env, group_map=None)

    env = TransformedEnv(
        env,
        Compose(
            DeviceCastTransform(
                device=args.device,
                in_keys=[("player", "observation"), ("player", "reward")],
            ),
            RewardSum(
                in_keys=[("player", "reward")],
                out_keys=[("player", "episode_reward")]
            ),
        )
    )


    print(env.observation_spec["player", "observation"].shape[
                -1
            ])
    check_env_specs(env) 

    return env