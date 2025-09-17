from gfootball.env import create_environment

env = create_environment(
    env_name='academy_3_vs_1_with_keeper',
    representation='simple115v2',
    render=True,
    number_of_left_players_agent_controls=3,
    write_full_episode_dumps=True,
    write_video=True,
    logdir="./traces",
)

obs = env.reset()
done = False

while not done:
    actions = list(env.action_space.sample())
    obs, reward, terminated, truncated, info = env.step(actions)

env.close()