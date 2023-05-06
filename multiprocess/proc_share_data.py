#!/usr/bin/env python

import sys
sys.path.append("..")

from multiprocessing import Process

import yaml
import tensorflow as tf
import gymnasium as gym
from tensorflow.keras.utils import to_categorical

from agents.dqn_model import Agent

def get_copier_embedding(copier, observation, num_actions):
    if copier is None:
        copier_embedding = np.zeros(num_actions)
    else:
        copier_prediction = copier.predict(observation)
        copier_embedding = to_categorical(copier_prediction, num_classes=num_actions)
    return copier_embedding

def agent(idx, dqn_config, num_episodes, copier):
    env = gym.make('Taxi-v3')
    observation, _ = env.reset()
    num_actions = 6 # taxi
    obs_dim = 500 # taxi
    assert(len(observation) == obs_dim)
    dqn_agent = Agent(id=idx,
                      input_dims=obs_dim + num_actions,
                      n_actions=num_actions,
                      **dqn_config)
    for episode in range(num_episodes):
        observation, reward, termination, truncation, info = env.last() 
        if termination or truncation:
            break

        copier_embedding = get_copier_embedding(copier, observation, num_actions)
        observation_with_copier_embedding = np.concatenate((observation, copier_embedding))
        action, dqn_command, entropy = dqn_agent.choose_action(observation_with_copier_embedding)
        env.step(action_i)

        new_observation, reward, termination, truncation, info = env.last()
        new_copier_embedding = get_copier_embedding(copier, new_observation, num_actions)
        new_observation_with_copier_embedding = np.concatenate((new_observation, new_copier_embedding))
        dqn_agent.store_transition(observation_with_copier_embedding, action, reward, new_observation_with_copier_embedding, False)

        if episode > episodes_until_dqn_learn:
            dqn_agent.learn()

def main():
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    num_agents = 4
    processes = list()
    for idx in range(num_agents):
        p = Process(target=agent, args=(idx, config["dqn_config"]))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
