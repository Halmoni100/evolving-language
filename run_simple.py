#from pettingzoo.mpe import simple_tag_v2
import sys
from pettingzoo.mpe import simple_v2
from agents.dqn_model import Agent
from agents.copier import Copier
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np


OBS_DIM = 4
NUM_ACTIONS = 5
COPIER_EP_LOOKBACK = 20

ACTION2EMBEDDING = {
    0: np.array([0, 0]),
    1: np.array([0, 1]),
    2: np.array([1, 0]),
    3: np.array([0, -1]),
    4: np.array([-1, 0]),
}

def main():
    #env = simple_tag_v2.env(render_mode='human')
    env = simple_v2.env(render_mode='human', max_cycles=200)

    env.reset()

    episodes = 200 
    agent_list = [] 
    lr = 0.005

    for i in range(env.num_agents): 
        agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=OBS_DIM+2, n_actions=5, mem_size=5000, batch_size=64, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")

        agent_list.append(agent_dqn)

    # [COPIER] INITIALIZE COPIER
    copier = Copier(env.num_agents, COPIER_EP_LOOKBACK, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)


    for ep_i in range(episodes): 

        done_n = [False for _ in range(env.num_agents)]
        ep_reward = 0 

        #env.seed(ep_i)
        env.reset(seed=ep_i)

        # [COPIER] TRAIN COPIER AT THE START OF EACH EPISODE
        copier.train()
        
        while not all(done_n): 
            

            for agent_i in range(env.num_agents):
                obs_i, reward_i, termination, truncation, info = env.last() 

                # [COPIER] GET COPIER PREDICTION + EMBED
                copier_prediction = copier.predict(obs_i)
                copier_prediction = ACTION2EMBEDDING[copier_prediction]

                # [COPIER] APPEND COPIER PREDICTION TO OBS_I
                obs_i_withcopier = np.concatenate((obs_i, copier_prediction))


                done_n[agent_i] = termination or truncation
                if termination or truncation: 
                    action_i = None
                    continue
                else: 
                    action_i = agent_list[agent_i].choose_action(obs_i_withcopier)
                    action_i = action_i[0]

                #print(action_i)
                env.step(action_i)


                # [COPIER] STORE OBS_I AND ACTION_I INTO COPIER BUFFER
                copier.store_obs_action(obs_i, action_i)

                # [COPIER] PREDICT FOR NEW STATE TO STORE INTO AGENT
                new_obs_i, reward_i, termination, truncation, info = env.last() 
                copier_prediction = copier.predict(new_obs_i)
                copier_prediction = ACTION2EMBEDDING[copier_prediction]
                new_obs_i_withcopier = np.concatenate((new_obs_i, copier_prediction))

                ep_reward += reward_i
                #print(obs_i)
                agent_list[agent_i].store_transition(obs_i_withcopier, action_i, reward_i, new_obs_i_withcopier, done_n[i])
                agent_list[agent_i].learn()


        print('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))
        with open('results_simple_wcopier.txt', 'a') as f:
            f.write('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))

    env.close()

