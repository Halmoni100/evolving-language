from pettingzoo.mpe import simple_tag_v2
from agents.dqn_model import Agent


env = simple_tag_v2.env(render_mode='human')

env.reset()

episodes = 200
agent_list = [] 
lr = 0.001

for i in range(env.num_agents): 
    agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=4, n_actions=5, mem_size=50000, batch_size=64, epsilon_dec=1, epsilon_end=0.001, fname="dqn_model_23jul.h5")

    agent_list.append(agent_dqn)


for ep_i in range(episodes): 

    done_n = [False for _ in range(env.num_agents)]
    ep_reward = 0

    #env.seed(ep_i)
    env.reset(seed=ep_i)
    

    while not all(done_n): 

        for agent_i, agent in enumerate(env.agent_iter()):
            obs_i, reward_i, termination, truncation, info = env.last() 

            if termination or truncation: 
                action_i = None
            else: 
                action_i = agent_list[agent_i].choose_action(obs_i)
            #print(action_i)
            env.step(action_i[0])
            new_obs_i, reward_i, termination, truncation, info = env.last() 

            ep_reward += reward_i
            print(obs_i)
            agent_list[agent_i].store_transition(obs_i, action_i, reward_i, new_obs_i, termination)
            agent_list[agent_i].learn()

        with open('results.txt', 'a') as f:
            f.write('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))

env.close()

