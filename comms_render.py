#from pettingzoo.mpe import simple_tag_v2
from pettingzoo.mpe import simple_v2, simple_world_comm_v2
from agents.dqn_model import Agent


#env = simple_tag_v2.env(render_mode='human')
#  env = simple_v2.env(render_mode='human', max_cycles=200)

num_food = 1
env = simple_world_comm_v2.env(max_cycles=200, render_mode='human',
        num_good=1, num_adversaries=1, 
        num_obstacles=0, num_food=num_food, num_forests=0)

env.reset()

episodes = 200
agent_list = [] 
lr = 0.005

leader_obs_dim = 12 + num_food * 2
#leader_obs_dim = 6
good_agent_obs_dim = 6 + num_food * 2

for i in range(env.num_agents): 

    if i == 0: # leader
        agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, 
                input_dims=leader_obs_dim, n_actions=20, 
                mem_size=5000, batch_size=64, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")
    else: # good agent
        agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, 
                input_dims=good_agent_obs_dim, n_actions=5, 
                mem_size=5000, batch_size=64, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")

    agent_list.append(agent_dqn)


for ep_i in range(episodes): 

    done_n = [False for _ in range(env.num_agents)]
    old_obs = [None for _ in range(env.num_agents)] 
    old_action = [None for _ in range(env.num_agents)] 
    ep_reward = 0

    #env.seed(ep_i)
    env.reset(seed=ep_i)
    

    while not all(done_n): 

	
        for agent_i in range(env.num_agents):
            obs_i, reward_i, termination, truncation, info = env.last() 

            #print("agent_i: ", agent_i)
            #print("obs_i.shape: ", obs_i.shape)
            if old_obs[agent_i] is not None: 
                agent_list[agent_i].store_transition(old_obs[agent_i], old_action[agent_i], reward_i, obs_i, done_n[i])
                agent_list[agent_i].learn()
                ep_reward += reward_i


            done_n[agent_i] = termination or truncation
            if termination or truncation: 
                action_i = None
                break
            else: 
                action_i = agent_list[agent_i].choose_action(obs_i)
                action_i = action_i[0]

            old_obs[agent_i] = obs_i
            old_action[agent_i] = action_i
            env.step(action_i)

            #new_obs_i, reward_i, termination, truncation, info = env.last() 
            #print("new_obs_i.shape: ", new_obs_i.shape)

    print('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))
    with open('results.txt', 'a') as f:
        f.write('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))

env.close()
