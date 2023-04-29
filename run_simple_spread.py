import numpy as np
import os
from pettingzoo.mpe import simple_spread_v2
from agents.dqn_model import Agent
from agents.copier import Copier
from utils import plot_rewards

"""
Environment: simple_spread_v2
https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/mpe/simple_spread/simple_spread.py
"""

OBS_DIM = 18
NUM_ACTIONS = 5
COPIER_EP_LOOKBACK = 20
NUM_EPISODES = 200
NUM_AGENTS = 3
MAX_CYCLES_PER_EP = 100
resultdir = r'/Users/eleanorye/Documents/GitHub/evolving-language/results'

ACTION2EMBEDDING = {
    0: np.array([0, 0]),
    1: np.array([0, 1]),
    2: np.array([1, 0]),
    3: np.array([0, -1]),
    4: np.array([-1, 0]),
}
COPIER_ACTION_EMBED_DIM = 2


def main(copier:bool):
    if copier:
        result_filename = 'results_simple_spread_copier.txt'
    else:
        result_filename = 'results_simple_spread.txt'

    env = simple_spread_v2.env(N=NUM_AGENTS, 
                               local_ratio=0.5, 
                               max_cycles=MAX_CYCLES_PER_EP, 
                               continuous_actions=False,)
    env.reset()

    agent_list = [] 
    lr = 0.005
    for i in range(env.num_agents): 
        if copier:
            input_dims = OBS_DIM+COPIER_ACTION_EMBED_DIM
        else:
            input_dims = OBS_DIM
        agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=input_dims, n_actions=NUM_ACTIONS, mem_size=5000, batch_size=64, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")
        agent_list.append(agent_dqn)
    print("Agents initialized!")

    # [COPIER] INITIALIZE COPIER
    if copier:
        copier = Copier(env.num_agents, COPIER_EP_LOOKBACK, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)
        print("Copier initialized!")


    for ep_i in range(NUM_EPISODES): 
        done_n = [False for _ in range(env.num_agents)]
        ep_reward = 0 

        #env.seed(ep_i)
        env.reset(seed=ep_i)

        # [COPIER] TRAIN COPIER AT THE START OF EACH EPISODE
        if copier:
            copier.train()
        
        while not all(done_n): 
            

            for agent_i in range(env.num_agents):
                obs_i, reward_i, termination, truncation, info = env.last() 

                # [COPIER] GET COPIER PREDICTION + EMBED
                if copier:
                    copier_prediction = copier.predict(obs_i)
                    copier_prediction = ACTION2EMBEDDING[copier_prediction]
                    # [COPIER] APPEND COPIER PREDICTION TO OBS_I
                    obs_i_withcopier = np.concatenate((obs_i, copier_prediction))

                done_n[agent_i] = termination or truncation
                if termination or truncation: 
                    action_i = None
                    continue
                else: 
                    if copier:
                        action_i = agent_list[agent_i].choose_action(obs_i_withcopier)
                    else:
                        action_i = agent_list[agent_i].choose_action(obs_i)

                    action_i = action_i[0]

                #print(action_i)
                env.step(action_i)


                # [COPIER] STORE OBS_I AND ACTION_I INTO COPIER BUFFER
                if copier:
                    copier.store_obs_action(obs_i, action_i)

                
                new_obs_i, reward_i, termination, truncation, info = env.last() 

                # [COPIER] PREDICT FOR NEW STATE TO STORE INTO AGENT
                if copier:
                    copier_prediction = copier.predict(new_obs_i)
                    copier_prediction = ACTION2EMBEDDING[copier_prediction]
                    new_obs_i_withcopier = np.concatenate((new_obs_i, copier_prediction))

                ep_reward += reward_i
                #print(obs_i)
                if copier:
                    agent_list[agent_i].store_transition(obs_i_withcopier, action_i, reward_i, new_obs_i_withcopier, done_n[i])
                else:
                    agent_list[agent_i].store_transition(obs_i, action_i, reward_i, new_obs_i, done_n[i])

                agent_list[agent_i].learn()


        print('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))
        with open(os.path.join(resultdir, result_filename), 'a') as f:
            f.write('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))

    env.close()

    plot_rewards(os.path.join(resultdir, result_filename))
    

