import numpy as np
import os
from pettingzoo.mpe import simple_spread_v2, simple_v2
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

ENV_NAME = 'simple_spread'
NUM_ACTIONS = 5
parentdir = r'/Users/eleanorye/Documents/GitHub/evolving-language/'
resultdir = os.path.join(parentdir, 'results')
checkpointsdir = os.path.join(parentdir, 'checkpoints', ENV_NAME)


ACTION2EMBEDDING = {
    0: np.array([0, 0]),
    1: np.array([0, 1]),
    2: np.array([1, 0]),
    3: np.array([0, -1]),
    4: np.array([-1, 0]),
}
COPIER_ACTION_EMBED_DIM = 2


def format_name(envname, use_copier, maxcycle, n_agents, copier_lb=None):
    if use_copier:
        name = '{}_copierlb{}_maxcycle{}_{}agent'.format(envname, copier_lb, maxcycle, n_agents)
    else:
        name = '{}_maxcycle{}_{}agent'.format(envname, maxcycle, n_agents)
    return name


def main(NUM_EPISODES: int,
         NUM_AGENTS: int,
         MAX_CYCLES_PER_EP: int, 
         use_copier:bool, 
         continue_from_checkpoint:bool, 
         lr:float=0.001,
         batch_size:int=64,
         copier_ep_lookback=None,
         checkpoint_ep=None
         ):

    OBS_DIM = NUM_AGENTS * 6  ## for simple_spread
    version_name = format_name(ENV_NAME, use_copier, MAX_CYCLES_PER_EP, NUM_AGENTS, copier_ep_lookback)
    result_filename = version_name + '_batch'+str(batch_size)+'.txt'
    print("Ep rewards saving to {}".format(result_filename))

    agent_list = [] 
    print("learning rate:", lr)
    print("batch size:", batch_size)
    if use_copier:
        input_dims = OBS_DIM+COPIER_ACTION_EMBED_DIM
    else:
        input_dims = OBS_DIM

    if not continue_from_checkpoint:  ## INITIALIZE EVERYTHING
        env = simple_spread_v2.env(N=NUM_AGENTS, 
                                local_ratio=0.5, 
                                max_cycles=MAX_CYCLES_PER_EP, 
                                continuous_actions=False,)
        
        #env = simple_v2.env(max_cycles=MAX_CYCLES_PER_EP)
        env.reset()
        
        for i in range(env.num_agents): 
            agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=input_dims, n_actions=NUM_ACTIONS, mem_size=5000, batch_size=batch_size, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")
            agent_list.append(agent_dqn)
        print("Agents initialized!")

        # [COPIER] INITIALIZE COPIER
        if use_copier:
            copier = Copier(env.num_agents, copier_ep_lookback, obs_dim=OBS_DIM, num_actions=NUM_ACTIONS)
            print("Copier initialized!")
        
        eps_to_train = range(NUM_EPISODES)
        global_ep = NUM_EPISODES
    
    else:
        package = load_checkpoint(version_name, checkpoint_ep)
        
        env = package['env']
        agent_paths = package['agent_paths']
        copier = package['copier']
        copier_nn_path = package['copier_nn_path']
        eps_to_train = range(package['ep_trained'], package['ep_trained']+NUM_EPISODES)
        global_ep = package['ep_trained']+NUM_EPISODES
        
        for i in range(len(agent_paths)):
            agent_dqn = Agent(gamma=0.998, epsilon=0.99, lr=lr, input_dims=input_dims, n_actions=NUM_ACTIONS, mem_size=5000, batch_size=batch_size, epsilon_dec=0.97, epsilon_end=0.001, fname="dqn_model_23jul.h5")
            agent_dqn.q_eval = load_model(package['agent_paths'][i])
            agent_dqn.memory = package["agent_memories"][i]
            agent_list.append(agent_dqn)
        if use_copier:
            copier.model = load_model(copier_nn_path)


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
                #agent_name = env.agents[agent_i]
                #reward_i = env.rewards[agent_name]

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


        # At the end of each episode;
        for agent_i in range(env.num_agents):
            agent_list[agent_i].epsilon_decay()
            
        #print('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))
        with open(os.path.join(resultdir, result_filename), 'a') as f:
            f.write('Episode #{} Reward: {}\n'.format(ep_i, ep_reward))


    # Save training checkpoint
    version_save_dir = os.path.join(checkpointsdir, version_name)
    if not os.path.isdir(version_save_dir):
        os.mkdir(version_save_dir)

    agent_memories = []
    agent_paths = dict(zip(range(NUM_AGENTS),[os.path.join(version_save_dir,'agent'+str(i)) for i in range(NUM_AGENTS)]))
    for i in range(len(agent_list)):
        agt = agent_list[i]
        agt.q_eval.save(agent_paths[i])
        agent_memories.append(agt.memory)

    if use_copier:
        copier_nn_path = os.path.join(version_save_dir,'copier_nn')
        copier.model.save(copier_nn_path)
        copier.model = None
        copier_to_save = copier
    else:
        copier_to_save = None
        copier_nn_path = None

    save_package = {
        "version_name": version_name,
        "env": env,
        "agent_paths": agent_paths,
        "agent_memories": agent_memories,
        "copier": copier_to_save,
        "copier_nn_path": copier_nn_path,
        "ep_trained": global_ep,
    }

    dump_checkpoint(save_package, version_name)
    env.close()
    plot_rewards(os.path.join(resultdir, result_filename))
    

def load_checkpoint(version_name, ep_trained):
    filename = "ep"+str(ep_trained)+".pickle"
    with open(os.path.join(checkpointsdir, version_name, filename), "rb") as filepath:
        package = pickle.load(filepath)
    return package

def dump_checkpoint(package, version_name):
    filename = "ep"+str(package['ep_trained'])+".pickle"
    with open(os.path.join(checkpointsdir, version_name, filename), "wb") as to_save:
        pickle.dump(package, to_save)
    print("checkpoint saved!")
    return

