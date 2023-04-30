import numpy as np
import os
from pettingzoo.mpe import simple_spread_v2
from agents.dqn_model import Agent
from agents.copier import Copier
from utils import plot_rewards
import pickle
from typing import Optional
from tensorflow.keras.models import load_model

"""
Environment: simple_spread_v2
https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/mpe/simple_spread/simple_spread.py
"""

OBS_DIM = 18
NUM_ACTIONS = 5
COPIER_EP_LOOKBACK = 20
NUM_AGENTS = 3
MAX_CYCLES_PER_EP = 100
resultdir = r'/Users/eleanorye/Documents/GitHub/evolving-language/results'
checkpointsdir = r'/Users/eleanorye/Documents/GitHub/evolving-language/checkpoints/simple_spread'


ACTION2EMBEDDING = {
    0: np.array([0, 0]),
    1: np.array([0, 1]),
    2: np.array([1, 0]),
    3: np.array([0, -1]),
    4: np.array([-1, 0]),
}
COPIER_ACTION_EMBED_DIM = 2


def main(NUM_EPISODES:int, use_copier:bool, continue_from_checkpoint:bool, checkpoint_ep=None):
    if use_copier:
        result_filename = 'simple_spread_copier.txt'
    else:
        result_filename = 'simple_spread.txt'

    agent_list = [] 
    lr = 0.005
    if use_copier:
        input_dims = OBS_DIM+COPIER_ACTION_EMBED_DIM
    else:
        input_dims = OBS_DIM

    if not continue_from_checkpoint:  ## INITIALIZE EVERYTHING
        env = simple_spread_v2.env(N=NUM_AGENTS, 
                                local_ratio=0.5, 
                                max_cycles=MAX_CYCLES_PER_EP, 
                                continuous_actions=False,)
        env.reset()
        
        for i in range(env.num_agents): 
            agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=input_dims, n_actions=NUM_ACTIONS, mem_size=5000, batch_size=64, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")
            agent_list.append(agent_dqn)
        print("Agents initialized!")

        # [COPIER] INITIALIZE COPIER
        if use_copier:
            copier = Copier(env.num_agents, COPIER_EP_LOOKBACK, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)
            print("Copier initialized!")
        
        eps_to_train = range(NUM_EPISODES)
    
    else:
        if use_copier:
            package = load_checkpoint("simple_spread_copier", checkpoint_ep)
        else:
            package = load_checkpoint("simple_spread", checkpoint_ep)
        
        env = package['env']
        agent_paths = package['agent_paths']
        copier = package['copier']
        copier_nn_path = package['copier_nn_path']
        eps_to_train = range(package['ep_trained'], package['ep_trained']+NUM_EPISODES)
        for i in range(len(agent_paths)):
            agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=input_dims, n_actions=NUM_ACTIONS, mem_size=5000, batch_size=64, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")
            agent_dqn.q_eval = load_model(package['agent_paths'][i])
            agent_dqn.memory = package["agent_memories"][i]
            agent_list.append(agent_dqn)
        if use_copier:
            copier.model = load_model(copier_nn_path)

    global_ep = package['ep_trained']+NUM_EPISODES

    for ep_i in eps_to_train: 
        done_n = [False for _ in range(env.num_agents)]
        ep_reward = 0 
        env.reset(seed=ep_i)

        # [COPIER] TRAIN COPIER AT THE START OF EACH EPISODE
        if use_copier:
            copier.train()
        
        while not all(done_n): 

            for agent_i in range(env.num_agents):
                obs_i, reward_i, termination, truncation, info = env.last()

                # [COPIER] GET COPIER PREDICTION + EMBED
                if use_copier:
                    copier_prediction = copier.predict(obs_i)
                    copier_prediction = ACTION2EMBEDDING[copier_prediction]
                    # [COPIER] APPEND COPIER PREDICTION TO OBS_I
                    obs_i_withcopier = np.concatenate((obs_i, copier_prediction))

                done_n[agent_i] = termination or truncation
                if termination or truncation: 
                    action_i = None
                    continue
                else: 
                    if use_copier:
                        action_i = agent_list[agent_i].choose_action(obs_i_withcopier)
                    else:
                        action_i = agent_list[agent_i].choose_action(obs_i)

                    action_i = action_i[0]

                #print(action_i)
                env.step(action_i)


                # [COPIER] STORE OBS_I AND ACTION_I INTO COPIER BUFFER
                if use_copier:
                    copier.store_obs_action(obs_i, action_i)

                new_obs_i, reward_i, termination, truncation, info = env.last() 
                ## TODO: FIX REWARDS - env.last() returns cumulative reward and not reward from last episode
                agent_name = env.agents[agent_i]
                reward_i = env.rewards[agent_name]

                # [COPIER] PREDICT FOR NEW STATE TO STORE INTO AGENT
                if use_copier:
                    copier_prediction = copier.predict(new_obs_i)
                    copier_prediction = ACTION2EMBEDDING[copier_prediction]
                    new_obs_i_withcopier = np.concatenate((new_obs_i, copier_prediction))

                ep_reward += reward_i
                #print(obs_i)
                if use_copier:
                    agent_list[agent_i].store_transition(obs_i_withcopier, action_i, reward_i, new_obs_i_withcopier, done_n[i])
                else:
                    agent_list[agent_i].store_transition(obs_i, action_i, reward_i, new_obs_i, done_n[i])

                agent_list[agent_i].learn()


        #print('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))
        with open(os.path.join(resultdir, result_filename), 'a') as f:
            f.write('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))


    # Save training checkpoint
    agent_memories = []
    agent_paths = dict(zip(range(NUM_AGENTS),[os.path.join(checkpointsdir,'agent'+str(i)) for i in range(NUM_AGENTS)]))
    for i in range(len(agent_list)):
        agt = agent_list[i]
        agt.q_eval.save(agent_paths[i])
        agent_memories.append(agt.memory)

    if use_copier:
        env_name = "simple_spread_copier"
        copier_nn_path = os.path.join(checkpointsdir,'copier_nn')
        copier.model.save(copier_nn_path)
        copier.model = None
        copier_to_save = copier
    else:
        env_name = "simple_spread"
        copier_to_save = None
        copier_nn_path = None

    save_package = {
        "env_name": env_name,
        "env": env,
        "agent_paths": agent_paths,
        "agent_memories": agent_memories,
        "copier": copier_to_save,
        "copier_nn_path": copier_nn_path,
        "ep_trained": global_ep,
    }

    dump_checkpoint(save_package)
    env.close()
    plot_rewards(os.path.join(resultdir, result_filename))
    

def load_checkpoint(env_name, ep_trained):
    filename = env_name+"_ep"+str(ep_trained)+".pickle"
    with open(os.path.join(checkpointsdir, filename), "rb") as filepath:
        package = pickle.load(filepath)
    return package

def dump_checkpoint(package):
    filename = package['env_name']+"_ep"+str(package['ep_trained'])+".pickle"
    with open(os.path.join(checkpointsdir, filename), "wb") as to_save:
        pickle.dump(package, to_save)
    print("checkpoint saved!")
    return
