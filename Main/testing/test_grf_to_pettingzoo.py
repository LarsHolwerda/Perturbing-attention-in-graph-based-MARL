from gfootball import gfootball_pettingzoo_v1
env = gfootball_pettingzoo_v1.parallel_env('academy_3_vs_1_with_keeper', representation='simplev1', number_of_left_players_agent_controls=2)
print(env.reset(seed=0)) 
print(env.step({'player_0':0, 'player_1':0}))

