#!/usr/bin/env python

import sys
sys.path.append("..")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import yaml
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

from taxi import taxi_observation_transform
from suppress_output import RedirectStdStreams
from progress_bar import ProgressBar
from agents.dqn_model import Agent

class TBReward:
    def __init__(self, tb_dir):
        os.makedirs(tb_dir, exist_ok=True)
        self.tb_writer = tf.summary.create_file_writer(tb_dir)

    def add_reward(self, reward, episode):
        with self.tb_writer.as_default():
            tf.summary.scalar(name="reward", data=reward, step=episode)

def plot_rewards(reward_buffer, plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(reward_buffer)
    plot_path = os.path.join(plot_dir, "rewards.png")
    plt.savefig(plot_path)
    plt.close()

def train_agent(dqn_config, dqn_misc, num_episodes, observation_transform):
    env = gym.make('Taxi-v3')
    observation, info = env.reset()
    observation = observation_transform(observation)
    num_actions = 6 # taxi
    obs_dim = 4 + 5 + 5 + 5 # taxi
    assert(len(observation) == obs_dim)
    dqn_agent = Agent(id=0,
                      input_dims=obs_dim,
                      n_actions=num_actions,
                      **dqn_config)

    pb = ProgressBar(num_episodes, length=50)
    pb.start(front_msg="episodes ")

    reward_buffer = list()
    tb_reward = TBReward("simple_results/tb")

    for episode in range(num_episodes):
        pb.update(front_msg="episodes ")

        observation, info = env.reset()
        observation = observation_transform(observation)
        curr_observation = observation
        curr_reward = None
        curr_termination = False
        curr_truncation = False
        curr_info = info
        episode_reward = 0

        # run episode
        while True:
            if curr_termination or curr_truncation:
                break

            curr_action, dqn_command, entropy = dqn_agent.choose_action(curr_observation, verbose=0)
            next_observation, next_reward, next_termination, next_truncation, next_info = env.step(curr_action)
            next_observation = observation_transform(next_observation)

            episode_reward += next_reward

            dqn_agent.store_transition(curr_observation, curr_action, next_reward, next_observation, False)

            if episode > dqn_misc["episodes_until_learn"]:
                devnull = open(os.devnull, 'w')
                with RedirectStdStreams(stdout=devnull, stderr=devnull): # hack to suppress output
                    dqn_agent.learn()
                devnull.close()

            curr_observation = next_observation
            curr_reward = next_reward
            curr_termination = next_termination
            curr_truncation = next_truncation
            curr_info = next_info
        
        reward_buffer.append(episode_reward)
        tb_reward.add_reward(episode_reward, episode)
        plot_rewards(reward_buffer, "simple_results")

        dqn_agent.epsilon_decay()

        keras.backend.clear_session()

    pb.reset()

def main():
    with open("simple_config.yml") as f:
        config = yaml.load(f, Loader=yaml.Loader)

    train_agent(config["dqn_config"], 
                config["dqn_misc"], 
                config["num_episodes"],
                taxi_observation_transform)

if __name__ == "__main__":
    main()
