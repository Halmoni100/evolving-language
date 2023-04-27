#from pettingzoo.mpe import simple_tag_v2
from pettingzoo.mpe import simple_v2
from dqn_model import Agent


#env = simple_tag_v2.env(render_mode='human')
env = simple_v2.env(max_cycles=200)

env.reset()

episodes = 200
agent_list = [] 
lr = 0.005

for i in range(env.num_agents): 
    agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=4, n_actions=5, mem_size=5000, batch_size=64, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")

    agent_list.append(agent_dqn)


for ep_i in range(episodes): 

    done_n = [False for _ in range(env.num_agents)]
    ep_reward = 0

    #env.seed(ep_i)
    env.reset()
    

    while not all(done_n): 

        for agent_i in range(env.num_agents):
            obs_i, reward_i, termination, truncation, info = env.last() 

            done_n[agent_i] = termination or truncation
            if termination or truncation: 
                action_i = None
                break
            else: 
                action_i = agent_list[agent_i].choose_action(obs_i)
                action_i = action_i[0]

            #print(action_i)
            env.step(action_i)
            new_obs_i, reward_i, termination, truncation, info = env.last() 

            ep_reward += reward_i
            #print(obs_i)
            agent_list[agent_i].store_transition(obs_i, action_i, reward_i, new_obs_i, done_n[i])
            agent_list[agent_i].learn()

    print('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))
    with open('results.txt', 'a') as f:
        f.write('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))

env.close()

