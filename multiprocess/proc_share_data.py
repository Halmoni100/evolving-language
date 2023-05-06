#!/usr/bin/env python

import sys
sys.path.append("..")

from multiprocessing import Process

import yaml
import tensorflow as tf
import gymnasium as gym

from agents.dqn_model import Agent

def agent(idx, dqn_config):
    env = gym.make('Taxi-v3')
    num_actions = 6 # taxi
    obs_dim = 500 # taxi
    copier_action_embed_dim = 2
    dqn_agent = Agent(id=idx,
                      input_dims=obs_dim + copier_action_embed_dim,
                      n_actions=num_actions,
                      **dqn_config)

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
